# secrets/

This directory is git-ignored. Place credential files here, e.g.:

- `vertex-sa.json` — Google Cloud service account JSON with role
  *Vertex AI User*. Path is referenced by the `GOOGLE_APPLICATION_CREDENTIALS`
  variable in `.env`.

NEVER commit anything from this directory. The `.gitignore` at the repo
root already excludes `service-account*.json` and `credentials*.json`,
but be careful with file names.
