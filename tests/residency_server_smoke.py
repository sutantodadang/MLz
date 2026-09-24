"""End-to-end gate for official bounded GGML residency on the normal server.

Starts MLz servers and checks, by named invariant:
  * greedy chat, completion, and streaming output with residency on equal the
    ordinary llama.cpp path (residency off);
  * two concurrent scheduler requests on the backed model match serial output;
  * weight invariants from /v1/residency/metrics (budget, zero uploads,
    balanced hooks and pins);
  * an injected weight-acquisition failure returns 503, leaves no pin or
    active node, and the next request produces the reference output again;
  * an undersized state budget is rejected at startup;
  * Ctrl+C / SIGINT shuts the backed server down with exit code 0.

usage: python tests/residency_server_smoke.py --exe zig-out/bin/MLz --model models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
"""

import argparse
import concurrent.futures
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}" + (f": {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)
    return condition


class Server:
    def __init__(self, exe, model, port, extra, env=None, log_name="server"):
        self.port = port
        self.log = open(f"residency_smoke_{log_name}.log", "w+b")
        args = [exe, model, "--server", "--host", "127.0.0.1", "--port", str(port),
                "--ctx", "512", "--threads", "4", "--temp", "0", "--seed", "42"] + extra
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        self.proc = subprocess.Popen(args, stdout=self.log, stderr=subprocess.STDOUT,
                                     env={**os.environ, **(env or {})}, **kwargs)

    def wait_ready(self, timeout=180):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                return False
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=1):
                    return True
            except OSError:
                time.sleep(0.5)
        return False

    def request(self, path, body=None, timeout=300):
        url = f"http://127.0.0.1:{self.port}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read().decode()
        except urllib.error.HTTPError as err:
            return err.code, err.read().decode()

    def stop(self, graceful=True, timeout=60):
        if self.proc.poll() is None:
            if graceful:
                self.proc.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
                # The accept loop checks the exit flag between connections.
                try:
                    socket.create_connection(("127.0.0.1", self.port), timeout=1).close()
                except OSError:
                    pass
            else:
                self.proc.kill()
        try:
            code = self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            code = None
        self.log.seek(0)
        output = self.log.read().decode(errors="replace")
        self.log.close()
        return code, output


def chat_body(stream=False):
    return {"messages": [{"role": "user", "content": "Name three primary colors."}],
            "max_tokens": 16, "temperature": 0, "seed": 42, "stream": stream}


def chat_text(server):
    status, body = server.request("/v1/chat/completions", chat_body())
    if status != 200:
        return status, body
    return status, json.loads(body)["choices"][0]["message"]["content"]


def completion_text(server):
    status, body = server.request("/v1/completions", {"prompt": "The capital of France is",
                                                      "max_tokens": 16, "temperature": 0, "seed": 42})
    if status != 200:
        return status, body
    return status, json.loads(body)["choices"][0]["text"]


def stream_text(server):
    status, body = server.request("/v1/chat/completions", chat_body(stream=True))
    if status != 200:
        return status, body
    text = []
    for line in body.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        delta = json.loads(line[6:])["choices"][0].get("delta", {})
        text.append(delta.get("content") or "")
    return status, "".join(text)


def metrics(server):
    status, body = server.request("/v1/residency/metrics")
    return json.loads(body) if status == 200 else None


def check_weight_invariants(prefix, m, budget_bytes):
    if not check(f"{prefix}: metrics endpoint available", m is not None):
        return
    w, h, a, f = m["weight"], m["hooks"], m["acquire"], m["failures"]
    check(f"{prefix}: weight budget reported", w["budget_bytes"] == budget_bytes, str(w))
    check(f"{prefix}: peak mapped <= budget", w["peak_mapped_bytes"] <= w["budget_bytes"], str(w))
    check(f"{prefix}: zero uploaded weight bytes", h["uploaded_weight_bytes"] == 0, str(h))
    check(f"{prefix}: hooks balanced", h["pre"] == h["post"] and h["active"] == 0, str(h))
    check(f"{prefix}: pins balanced", h["acquires"] == h["releases"] and a["open_pins"] == 0, str(h) + str(a))
    mem = m["memory"]
    check(f"{prefix}: memory plan bounds allocation",
          mem is not None and mem["allocated"]["state_bytes"] <= mem["planned"]["state_bytes"]
          and mem["allocated"]["compute_bytes"] <= mem["planned"]["compute_bytes"]
          and mem["allocated"]["host_weight_bytes"] == 0, str(mem))
    return f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=18090)
    parser.add_argument("--weight-budget-mib", type=int, default=4)
    parser.add_argument("--state-budget-mib", type=int, default=1024)
    args = parser.parse_args()
    budget = args.weight_budget_mib * 1024 * 1024
    residency = ["--residency", "--weight-budget-mib", str(args.weight_budget_mib),
                 "--state-budget-mib", str(args.state_budget_mib)]

    # Reference: ordinary llama.cpp path.
    ref = Server(args.exe, args.model, args.port, ["--ngl", "0"], log_name="reference")
    if not check("reference server starts", ref.wait_ready()):
        print(ref.stop(graceful=False)[1][-4000:])
        return 1
    _, ref_chat = chat_text(ref)
    _, ref_completion = completion_text(ref)
    ref.stop(graceful=False)
    print(f"reference chat={ref_chat!r} completion={ref_completion!r}")

    # Backed single-stream server.
    srv = Server(args.exe, args.model, args.port, residency, log_name="backed")
    if not check("backed server starts", srv.wait_ready()):
        print(srv.stop(graceful=False)[1][-4000:])
        return 1
    status, text = chat_text(srv)
    check("backed chat equals reference", status == 200 and text == ref_chat, f"{status} {text!r}")
    status, text = completion_text(srv)
    check("backed completion equals reference", status == 200 and text == ref_completion, f"{status} {text!r}")
    status, text = stream_text(srv)
    check("backed streaming equals reference", status == 200 and text == ref_chat, f"{status} {text!r}")
    status, text = chat_text(srv)
    check("backed repeated chat equals reference", status == 200 and text == ref_chat, f"{status} {text!r}")
    check_weight_invariants("backed", metrics(srv), budget)
    code, output = srv.stop(graceful=True)
    check("backed server graceful shutdown exits 0", code == 0, f"exit={code}\n{output[-2000:]}")

    # Backed continuous-batching server with concurrent requests.
    srv = Server(args.exe, args.model, args.port, residency + ["--max-concurrent", "2"], log_name="scheduler")
    if check("scheduler server starts", srv.wait_ready()):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: chat_text(srv), range(2)))
        check("concurrent scheduler chats equal reference",
              all(status == 200 and text == ref_chat for status, text in results), str(results))
        check_weight_invariants("scheduler", metrics(srv), budget)
    srv.stop(graceful=False)

    # Injected mid-graph failure must unwind and leave the server usable.
    for label, extra in (("single", []), ("scheduler", ["--max-concurrent", "2"])):
        srv = Server(args.exe, args.model, args.port, residency + extra,
                     env={"MLZ_RESIDENCY_INJECT_FAILURE": "40"}, log_name=f"inject_{label}")
        if not check(f"injection {label} server starts", srv.wait_ready()):
            srv.stop(graceful=False)
            continue
        status, body = srv.request("/v1/chat/completions", chat_body())
        check(f"injection {label}: failed request returns 503 residency_error",
              status == 503 and "residency_error" in body, f"{status} {body}")
        m = metrics(srv)
        f = check_weight_invariants(f"injection {label}", m, budget)
        if f is not None:
            check(f"injection {label}: failure classified", f["graphs"] == 1 and f["injected"] == 1
                  and f["last_kind"] == "injected" and f["last_reason"] != "", str(f))
        status, text = chat_text(srv)
        check(f"injection {label}: next request equals reference", status == 200 and text == ref_chat,
              f"{status} {text!r}")
        srv.stop(graceful=False)

    # Oversized non-weight plan is rejected before allocation.
    srv = Server(args.exe, args.model, args.port, ["--residency", "--weight-budget-mib",
                                                   str(args.weight_budget_mib), "--state-budget-mib", "1"],
                 log_name="state_reject")
    try:
        code = srv.proc.wait(timeout=180)
    except subprocess.TimeoutExpired:
        code = None
    _, output = srv.stop(graceful=False)
    check("state budget rejection exits non-zero at startup",
          code not in (None, 0) and "state budget exceeded" in output, f"exit={code}\n{output[-2000:]}")

    print(f"\n{len(FAILURES)} failed invariant(s)" + (": " + ", ".join(FAILURES) if FAILURES else ""))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
