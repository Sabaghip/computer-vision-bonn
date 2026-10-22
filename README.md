# Computer Vision — MA-INF 2201 (University of Bonn)

Coursework and exercise solutions for **MA-INF 2201 Computer Vision**, taught at the
**University of Bonn**. Maintained by **Mohamadhosein (Parsa) Sabaghi**, a student in the
course, as a running record of each exercise sheet across the semester.

> **For readers (human or AI) new to this repo:** this is a university assignment
> repository, not a production codebase. Each `assignmentN/` folder is one graded exercise
> sheet, containing the sheet's PDF, the starter template provided by the course staff, and
> my own implementation. Code is written to satisfy each sheet's constraints (only the
> packages listed in the template, Python 3.12, OpenCV 5.0.0, NumPy 2.3.3, Linux).

## About this repository

Computer Vision is a broad field spanning image formation, low-level pixel operations,
feature extraction, geometric vision, and learning-based methods. This repository tracks my
progression through that material one exercise sheet at a time. The table below is kept up
to date as new sheets are added — it doubles as an index of which CV topics are covered
where, so anyone skimming the repo (or an LLM asked about it) can see at a glance what's
been implemented.

## Assignments

| Sheet | Folder | Topic | Core CV Concepts | Status |
|---|---|---|---|---|
| 0 | [`assignment1/`](assignment1/) | Introduction to OpenCV | Image I/O & display, RGB→HSV color space, pixel-wise transforms (loop vs. vectorized), ROI patch extraction/placement, grayscale thresholding & masking, drawing primitives & borders | In progress |

<!--
When adding a new assignment:
1. Create `assignmentN/` with the sheet PDF, template, and your solution code.
2. Add a row to the table above (sheet number, folder, topic, concepts, status).
3. If the sheet introduces a new major CV area (e.g. feature detection, stereo,
   segmentation, deep learning), consider adding it under "Topics Covered" below.
-->

## Topics Covered So Far

- **Image fundamentals:** reading/writing images, color space conversions, pixel-level
  manipulation
- **Performance:** vectorized NumPy operations vs. explicit pixel loops

*(This list grows as later sheets are added — see the table above for the authoritative,
per-sheet breakdown.)*

## Tech Stack

- **Language:** Python 3.12
- **Core libraries:** OpenCV 5.0.0, NumPy 2.3.3
- **Platform:** Linux
- Each sheet restricts usage to only the packages listed in its own template — no
  additional third-party libraries are used beyond what's permitted.

## Repository Structure

```
assignments/
├── README.md              # this file
├── .gitignore
├── requirements.txt        # pinned deps shared by all sheets
├── assignment1/            # Sheet 0 — Introduction to OpenCV
│   ├── CV1_sheet_0.pdf      # original exercise sheet
│   ├── Sheet00_template.py  # staff-provided starter template
│   ├── bonn.jpeg             # sample image used by the exercises
│   └── README                # group members for this sheet
└── assignmentN/             # future sheets, added as the semester progresses
```

## Setup

A single virtual environment at the repo root is shared across all assignment sheets.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This installs the exact versions required by the course (`numpy==2.3.3`,
`opencv-python==5.0.0.93`, matching OpenCV 5.0.0 / Python 3.12).

## Running the Code

Each assignment folder is self-contained. With the venv above activated, from inside a given
`assignmentN/` folder:

```bash
python SheetNN_<name>.py
```

## Academic Integrity

Per course policy, all submitted code must be original and understood by the author. This
repository reflects my own work (or joint work with my assigned teammate, where the course
permits group submission) — it is a personal record of coursework, not a reference to copy
from.
