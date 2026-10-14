#!/usr/bin/env python3
"""
Jules Bridge: Autonomous Agent Interface for Google Jules
Enables Antigravity and terminal workflows to dispatch tasks to Google Jules asynchronously.
"""

import os
import sys
import json
import argparse
import urllib.request
import urllib.error

API_BASE = "https://jules.googleapis.com/v1alpha"

def get_api_key():
    key = os.environ.get("JULES_API_KEY")
    if not key:
        raise ValueError("JULES_API_KEY environment variable is not set.")
    return key

def make_request(endpoint, method="GET", payload=None):
    url = f"{API_BASE}/{endpoint.lstrip('/')}"
    headers = {
        "X-Goog-Api-Key": get_api_key(),
        "Content-Type": "application/json"
    }
    data = json.dumps(payload).encode("utf-8") if payload else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    
    try:
        with urllib.request.urlopen(req) as resp:
            content = resp.read().decode("utf-8")
            return json.loads(content) if content else {}
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8")
        try:
            err_json = json.loads(err_msg)
            print(f"[Error {e.code}] {json.dumps(err_json, indent=2)}", file=sys.stderr)
        except Exception:
            print(f"[Error {e.code}] {err_msg}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[Network Error] {e}", file=sys.stderr)
        sys.exit(1)

def list_repos():
    res = make_request("sources")
    sources = res.get("sources", [])
    if not sources:
        print("No connected repositories found in Jules.")
        return
    print(f"Connected Repositories ({len(sources)}):")
    for s in sources:
        gh = s.get("githubRepo", {})
        owner = gh.get("owner", "")
        repo = gh.get("repo", "")
        default_branch = gh.get("defaultBranch", {}).get("displayName", "main")
        print(f"  - {owner}/{repo} (default: {default_branch})")

def list_sessions():
    res = make_request("sessions")
    sessions = res.get("sessions", [])
    if not sessions:
        print("No Jules sessions currently active or found.")
        return
    print(f"Sessions ({len(sessions)}):")
    for s in sessions:
        sid = s.get("name", "").split("/")[-1]
        title = s.get("title", "Untitled")
        state = s.get("state", "UNKNOWN")
        create_time = s.get("createTime", "")
        print(f"  [{sid}] {title} | Status: {state} | Created: {create_time}")

def get_session(session_id):
    sid = session_id.split("/")[-1]
    res = make_request(f"sessions/{sid}")
    print(json.dumps(res, indent=2))

def start_session(repo, prompt, title=None, branch=None):
    if "/" not in repo:
        repo = f"thejfisher/{repo}"
    
    # Check default branch if not provided
    if not branch:
        sources_res = make_request("sources")
        for s in sources_res.get("sources", []):
            gh = s.get("githubRepo", {})
            if f"{gh.get('owner')}/{gh.get('repo')}".lower() == repo.lower():
                branch = gh.get("defaultBranch", {}).get("displayName", "main")
                break
    branch = branch or "main"
    
    payload = {
        "prompt": prompt,
        "sourceContext": {
            "source": f"sources/github/{repo}",
            "githubRepoContext": {
                "startingBranch": branch
            }
        }
    }
    if title:
        payload["title"] = title
    else:
        payload["title"] = prompt[:50] + ("..." if len(prompt) > 50 else "")

    print(f"Dispatching task to Jules for {repo} (branch: {branch})...")
    res = make_request("sessions", method="POST", payload=payload)
    sid = res.get("name", "").split("/")[-1]
    print(f"Successfully dispatched!")
    print(f"Session ID: {sid}")
    print(f"Title: {res.get('title')}")
    print(f"State: {res.get('state', 'QUEUED/RUNNING')}")
    return res

def main():
    parser = argparse.ArgumentParser(description="Jules Bridge CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Repos
    subparsers.add_parser("repos", help="List connected repositories")

    # Sessions
    subparsers.add_parser("sessions", help="List active and recent Jules sessions")

    # Get session
    p_get = subparsers.add_parser("status", help="Get status of a session")
    p_get.add_argument("session_id", help="Session ID")

    # Start session
    p_start = subparsers.add_parser("start", help="Start a new Jules task/session")
    p_start.add_argument("--repo", "-r", required=True, help="Repo in format owner/repo or repo")
    p_start.add_argument("--prompt", "-p", required=True, help="Instruction prompt for Jules")
    p_start.add_argument("--title", "-t", help="Title for the session")
    p_start.add_argument("--branch", "-b", help="Starting branch (default: repo default)")

    args = parser.parse_args()

    if args.command == "repos":
        list_repos()
    elif args.command == "sessions":
        list_sessions()
    elif args.command == "status":
        get_session(args.session_id)
    elif args.command == "start":
        start_session(args.repo, args.prompt, args.title, args.branch)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
