## User Wiki

Durable user preferences and work rules live in `C:\Users\KN\.codex\user-wiki`.

For non-trivial work, first read `C:\Users\KN\.codex\user-wiki\AGENTS.md` and follow it.
Read only the related user-wiki pages beyond that.

Current user requests, system/developer instructions, and project-local `AGENTS.md` files take precedence over the user wiki.

## Paper Graph

The paper graph lives in `docs/paper/graph`.

Write graph nodes in Korean so the user can read them directly.

Keep graph filenames as short English nouns when possible, such as `background.md`, `claim.md`, and `method.md`.

Each graph node should be independently readable, but the voice should feel consistent across nodes.
Use the tone already established in `background.md` and `claim.md`.

Use concrete recurring terms consistently:

- 2D 조각 이미지
- 3D 후보
- 목표 수치
- 패턴 수치

Start each node with the core point of that node.
Explain the concrete role first, then attach technical terms only when needed.

Do not write as if the reader should look something up later.
The document should be understandable while reading it.

Strong factual claims need evidence.
Put numeric references near the claim they support, and make those references jump to the source.

Use full vault-path wikilinks for graph-node navigation, such as `[[docs/paper/graph/claim|논문의 주장]]`.
Avoid bare alias wikilinks like `[[claim|논문의 주장]]` and bare markdown links like `[논문의 주장](claim.md)` for graph-node navigation because some viewers can open a blank same-named page outside `docs/paper/graph`.

When a planned graph node does not exist yet, mention it as text instead of creating a broken wikilink.

Before finishing graph edits, check:

- graph-node links use the `docs/paper/graph/...` vault path and point to existing graph nodes
- markdown links point to existing files or valid anchors
- image links point to existing assets
- reference ids used in text exist in the same file
- the main reading path in `docs/paper/graph/index.md` does not point to missing nodes

## Project Goal

The project goal lives in `docs/project/goal.md`.

Keep this separate from the paper graph. The paper graph explains MicroLad as written. The project goal describes the extension this repository is trying to build on top of MicroLad.

The project aims to implement MicroLad and add conditioning on actual observations. This includes one or more observed 2D slices at specified internal positions, scale-up from a provided 3D structure or external slice set, and large 2D crop conditioning such as using a 512x512 crop to generate a matching 512x512x512 candidate. The generated 3D candidate should satisfy those observation conditions while remaining realistic.
