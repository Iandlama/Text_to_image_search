# DLS_project

Text-to-image search over memes. Each meme is an image (`images/{id}.webp`)
paired with textual signals for retrieval in `metadata.jsonl`.

## Dataset Structure (`metadata.jsonl`)

One JSON object per line, one line per meme:

* **`id`** — MD5 hash of the image, also used as its filename (`images/{id}.webp`).
* **`title`** — title of the post the meme comes from.
* **`image_desc`** — what is depicted (visual signal, matches visual queries).
* **`meaning`** — the meaning/joke of the meme (semantic signal, matches intent queries).

Example:

```json
{"id": "c343ad68...", "title": "And that's a fact", "image_desc": "Two dogs carry a white flag ...", "meaning": "Meme poster is saying that searching Google ..."}
```

## Scripts

* `loader/meme_loader.py` — downloads the dataset from Hugging Face and packs images + `metadata.jsonl` into a zip.
* `loader/meme_utils.py` — `parse_caption()`: parses the raw memecap prompt into `title` / `image_desc` / `meaning` fields.
* `loader/migrate_metadata.py` — rebuilds `metadata.jsonl` inside an already-downloaded archive to the new schema (no re-download).
