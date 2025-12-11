import os
import json
import argparse


def load_json_file(path: str):
    """
    Load .json or .jsonl safely.

    .json  → single JSON object
    .jsonl → list of JSON objects (one per line)
    """
    try:
        if path.endswith(".jsonl"):
            with open(path, "r") as f:
                return [json.loads(line) for line in f if line.strip()]

        # Standard JSON
        with open(path, "r") as f:
            return json.load(f)

    except Exception as e:
        return f"ERROR_READING_JSON({path}): {str(e)}"


def combine_jsons_recursive(folder: str):
    """
    Recursively walk a folder and load ALL .json / .jsonl files.

    Output structure:
    {
        "subdir1": {...},
        "sampler.jsonl": [...],
        "config.json": {...}
    }
    """
    result = {}

    for entry in sorted(os.listdir(folder)):
        full_path = os.path.join(folder, entry)

        # Subfolder → recurse
        if os.path.isdir(full_path):
            result[entry] = combine_jsons_recursive(full_path)

        # JSON or JSONL
        elif entry.lower().endswith((".json", ".jsonl")):
            result[entry] = load_json_file(full_path)

        # Ignore others
        else:
            continue

    return result


def load_single_code_file(file_path: str):
    try:
        with open(file_path, "r") as f:
            return {os.path.basename(file_path): f.read()}
    except Exception as e:
        return {os.path.basename(file_path): f"ERROR_READING_FILE: {str(e)}"}


def load_code_files_recursive(folder: str, prefix=""):
    """
    Recursively load all .py files under folder.

    Result:
    {
        "train.py": "...",
        "subdir/utils.py": "...",
    }
    """
    code_dict = {}

    for entry in sorted(os.listdir(folder)):
        full_path = os.path.join(folder, entry)
        rel_name = os.path.join(prefix, entry) if prefix else entry

        if os.path.isdir(full_path):
            code_dict.update(load_code_files_recursive(full_path, rel_name))

        elif entry.endswith(".py"):
            try:
                with open(full_path, "r") as f:
                    code_dict[rel_name] = f.read()
            except Exception as e:
                code_dict[rel_name] = f"ERROR_READING_FILE: {str(e)}"

    return code_dict


def load_code(path: str):
    """
    Detect whether the input is a file or directory.
    """
    if os.path.isfile(path):
        print(f"[INFO] Loading single python file: {path}")
        return load_single_code_file(path)
    else:
        print(f"[INFO] Loading python files in directory: {path}")
        return load_code_files_recursive(path)


def export_summary(log_folder: str, code_path: str, out_path: str):
    print(f"[INFO] Reading logs from: {log_folder}")
    logs = combine_jsons_recursive(log_folder)

    print(f"[INFO] Reading code from: {code_path}")
    code = load_code(code_path)

    combined = {
        "logs": logs,
        "code": code
    }

    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"[OK] Combined summary written to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine TraceML JSON logs + Python code into one file for LLM analysis."
    )

    parser.add_argument(
        "logs",
        type=str,
        help="Path to folder containing TraceML log files (JSON / JSONL)"
    )

    parser.add_argument(
        "code",
        type=str,
        help="Path to python code: either a .py file or directory containing .py files"
    )

    parser.add_argument(
        "--out",
        type=str,
        default="combined_summary.json",
        help="Output JSON file"
    )

    args = parser.parse_args()

    export_summary(args.logs, args.code, args.out)
