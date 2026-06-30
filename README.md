# DLS_project

## Dataset Structure (`metadata.jsonl`)

The primary index file linking text data to the images:

* **`id`** — Unique MD5 hash of the image, used as the filename (`{hash}.webp`).
* **`caption[0] (user)`** — Visual description (what is depicted).
* **`caption[1] (assistant)`** — Semantic meaning (the joke and context).