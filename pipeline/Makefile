# =============================================================================
# Cell-motility pipeline   movies -> tracks -> per-cell MSD
# =============================================================================
# Everything runs in THIS folder. Put your *.avi movies here (or in subfolders)
# next to these files, then run `make`. Two stages run in order:
#
#   stage1_trackmate.py    *.avi               -> results/<stem>_spots.csv (+ overlays)   [Fiji/TrackMate]
#   stage2_compute_msd.py  results/*_spots.csv -> msd_per_cell/<stem>_msd.csv             [python3]
#
# Quick start
# -----------
#   make                     # run both stages on movies in this folder
#   make print               # show detected movies + every setting + Fiji path
#   make track               # stage 1 only  (movies -> spots CSVs + overlays)
#   make msd                 # full pipeline (same as `make`)
#   make clean               # delete results/ msd_per_cell/
#   make install-fiji        # optional: download Fiji (~400 MB) into
#                            #   ~/Applications (override FIJI_INSTALL_DIR)
#
# Fiji is autodetected in the usual install locations. If yours lives somewhere
# unusual, pass it once:
#   make FIJI=/Applications/Fiji.app/Contents/MacOS/fiji-macos-arm64
#
# All detector/tracker/MSD parameters are overridable variables. See README.md
# for the full table; `make print` shows the current values.
# =============================================================================

SHELL := /bin/bash

# ---- scripts (run order shown by the stageN_ prefix) -----------------------
STAGE1 := stage1_trackmate.py
STAGE2 := stage2_compute_msd.py
GATHER := gather_movies.sh

# ---- output directories ----------------------------------------------------
SPOTS_DIR := results
MSD_DIR   := msd_per_cell

# ---- Python interpreter (needs pandas + numpy + matplotlib) ----------------
PYTHON ?= python3

# =============================================================================
# IMAGING PROFILE
# =============================================================================
# The paper used two imaging configurations, and they do NOT share settings.
# Six parameters differ between them, so there is no single set of defaults
# that is correct for both. Pick the profile that matches your data:
#
#   make PROFILE=phase    phase contrast -- density + PMA datasets (Figs 4b, 4c)
#   make PROFILE=gfp      GFP            -- dox titration dataset  (Fig 6b)
#
# Values below were read directly from the TrackMate XMLs that produced the
# published data (TrackMate 8.1.6), not from memory. See VERSIONS_AND_PROVENANCE.
#
# The two profiles are one physical rule applied to two configurations: the GFP
# pixel is exactly 1.5x the phase pixel, and phase images every 180 s vs GFP's
# 120 s, so in micrometers the phase distances sit a consistent ~1.22x above
# GFP's. If you add a third configuration, scale it the same way rather than
# copying either profile verbatim.
# =============================================================================
PROFILE ?= phase

ifeq ($(PROFILE),phase)
  DIAMETER                 ?= 36.0      # TrackMate RADIUS 18.0 px
  LINKING_MAX_DISTANCE     ?= 46.0
  GAP_CLOSING_MAX_DISTANCE ?= 55.0
  UM_PER_PIXEL             ?= 0.68626
  LAG                      ?= 3         # 3 frames x 180 s = 540 s
  FRAME_INTERVAL_S         ?= 180
else ifeq ($(PROFILE),gfp)
  DIAMETER                 ?= 20.0      # TrackMate RADIUS 10.0 px
  LINKING_MAX_DISTANCE     ?= 25.0
  GAP_CLOSING_MAX_DISTANCE ?= 30.0
  UM_PER_PIXEL             ?= 1.02939
  LAG                      ?= 5         # 5 frames x 120 s = 600 s
  FRAME_INTERVAL_S         ?= 120
else
  $(error PROFILE must be 'phase' or 'gfp' (got '$(PROFILE)'))
endif

# ---- Stage 1: LoG detector settings ----------------------------------------
# The GUI shows "Diameter"; TrackMate stores RADIUS = DIAMETER/2 (the script
# does the conversion). QUALITY is the initial-spot QUALITY filter, which is
# separate from the detector THRESHOLD.
# These were identical across both channels in the published XMLs.
QUALITY          ?= 0.0
TARGET_CHANNEL   ?= 1
THRESHOLD        ?= 0.2
MEDIAN_FILTERING ?= true
SUBPIXEL         ?= true

# ---- Stage 1: tracker settings ---------------------------------------------
# TRACKER selects one of two mutually exclusive trackers:
#   simple = Simple LAP  (linking / gap-closing / frame-gap only)
#   lap    = Sparse LAP  (adds gap-closing toggle + splitting / merging)
# The ALLOW_*/SPLITTING/MERGING knobs are ignored when TRACKER=simple.
# The paper used Sparse LAP for both channels, with splitting and merging off.
TRACKER                  ?= lap
MAX_FRAME_GAP            ?= 2
ALLOW_GAP_CLOSING        ?= true
ALLOW_SPLITTING          ?= false
SPLITTING_MAX_DISTANCE   ?= 15.0
ALLOW_MERGING            ?= false
MERGING_MAX_DISTANCE     ?= 15.0

# ---- Stages 2 & 3: MSD settings --------------------------------------------
# NOTE: MIN_CONSEC is NOT yet verified against the published analysis. 18 is
# carried over from the original pipeline. Phase movies are 20 frames and GFP
# movies are 30, so a single threshold is not obviously right for both --
# confirm before relying on the "complete" MSD variants.
MIN_CONSEC   ?= 18

# ---- Fiji autodetection ----------------------------------------------------
# Searches common install roots for any launcher name Fiji has used over time.
# Override by passing FIJI=/path/to/launcher on the command line.
FIJI_BASE_DIRS := \
  /Applications $(HOME)/Applications $(HOME)/Desktop $(HOME)/Downloads $(HOME) .
FIJI_LAUNCHER_NAMES := \
  fiji-macos-arm64 fiji-macos-x86_64 Fiji ImageJ-macosx
FIJI_CANDIDATES := \
  $(foreach base,$(FIJI_BASE_DIRS),$(base)/Fiji/fiji) \
  $(foreach base,$(FIJI_BASE_DIRS),\
    $(foreach name,$(FIJI_LAUNCHER_NAMES),\
      $(base)/Fiji.app/Contents/MacOS/$(name) \
      $(base)/Fiji/$(name)))
FIJI ?= $(firstword $(foreach p,$(FIJI_CANDIDATES),$(wildcard $(p))))

# ---- input discovery -------------------------------------------------------
# Only the top level is scanned here; nested layouts are handled by the gather
# step in `all` (which copies subfolder movies up first). Our own overlay
# outputs (*_tracks.avi) are excluded.
AVIS  := $(filter-out %_tracks.avi,$(wildcard *.avi))
STEMS := $(AVIS:.avi=)

SPOTS_CSVS := $(addprefix $(SPOTS_DIR)/,$(addsuffix _spots.csv,$(STEMS)))
MSD_CSVS   := $(addprefix $(MSD_DIR)/,$(addsuffix _msd.csv,$(STEMS)))

# ---- Fiji install location (only used by install-fiji) ---------------------
# Deliberately NOT the current directory. Installing Fiji next to your data
# buries a ~400 MB application inside a project folder where Spotlight cannot
# see it, and makes it easy to later report the version of some *other* Fiji.
# Override if you want it elsewhere:  make FIJI_INSTALL_DIR=/Applications install-fiji
FIJI_INSTALL_DIR ?= $(HOME)/Applications

# ---- Fiji download URL (only used by install-fiji) -------------------------
FIJI_ARCH := $(shell uname -m)
ifeq ($(FIJI_ARCH),arm64)
FIJI_URL := https://downloads.imagej.net/fiji/latest/fiji-latest-macos-arm64-jdk.zip
else
FIJI_URL := https://downloads.imagej.net/fiji/latest/fiji-latest-macos-x86_64-jdk.zip
endif

.PHONY: all build track msd flatten clean check-fiji install-fiji \
        print diagnose help

# `all` gathers movies from subfolders if the top level is empty, then
# re-invokes `build` so discovery re-runs and sees the freshly gathered movies.
all:
	@if [ -z "$(strip $(AVIS))" ]; then \
	  echo "No *.avi at the top level of $(CURDIR)."; \
	  echo "Looking for movies in subfolders..."; \
	  bash "$(GATHER)" "$(CURDIR)" "$(SPOTS_DIR)"; \
	fi
	@$(MAKE) --no-print-directory build

build:
	@if [ -z "$(strip $(AVIS))" ]; then \
	  echo "Nothing to process: no *.avi here or in any subfolder."; \
	  echo "Drop your movies in this folder (or in subfolders) and re-run 'make'."; \
	else \
	  $(MAKE) --no-print-directory msd && \
	  echo "" && echo "Done. Per-cell MSD CSVs are in $(MSD_DIR)/"; \
	fi

# ---- stage targets ---------------------------------------------------------
track: $(SPOTS_CSVS)
msd:   $(MSD_CSVS)

# ---- stage 1: one movie -> spots CSV (+ overlays) --------------------------
# Multi-target pattern rule: a single Fiji call writes all four outputs per
# movie, so GNU Make (incl. 3.81 on stock macOS) runs the recipe once.
$(SPOTS_DIR)/%_spots.csv $(SPOTS_DIR)/%_tracks.avi $(SPOTS_DIR)/%_tracks.tif $(SPOTS_DIR)/%_trackmate.xml: %.avi $(STAGE1) | check-fiji $(SPOTS_DIR)
	@echo "==> [stage 1] TrackMate: $< (tracker=$(TRACKER) diameter=$(DIAMETER) threshold=$(THRESHOLD) quality=$(QUALITY))"
	"$(FIJI)" --headless \
		--run "$(STAGE1)" \
		"input='$(CURDIR)/$<',outdir='$(CURDIR)/$(SPOTS_DIR)',stem='$*',quality=$(QUALITY),target_channel=$(TARGET_CHANNEL),diameter=$(DIAMETER),threshold=$(THRESHOLD),median_filtering=$(MEDIAN_FILTERING),subpixel=$(SUBPIXEL),tracker='$(TRACKER)',linking_max_distance=$(LINKING_MAX_DISTANCE),gap_closing_max_distance=$(GAP_CLOSING_MAX_DISTANCE),max_frame_gap=$(MAX_FRAME_GAP),allow_gap_closing=$(ALLOW_GAP_CLOSING),allow_splitting=$(ALLOW_SPLITTING),splitting_max_distance=$(SPLITTING_MAX_DISTANCE),allow_merging=$(ALLOW_MERGING),merging_max_distance=$(MERGING_MAX_DISTANCE)"

# ---- stage 2: spots CSV -> per-cell MSD CSV --------------------------------
$(MSD_DIR)/%_msd.csv: $(SPOTS_DIR)/%_spots.csv $(STAGE2) | $(MSD_DIR)
	@echo "==> [stage 2] MSD: $<"
	$(PYTHON) "$(STAGE2)" "$<" -o "$@" \
	    --lag $(LAG) --min-consec $(MIN_CONSEC) --um-per-pixel $(UM_PER_PIXEL)

# ---- output directories ----------------------------------------------------
$(SPOTS_DIR) $(MSD_DIR):
	@mkdir -p "$@"

# ---- helpers ---------------------------------------------------------------
# Copy every *.avi from subfolders up to the top level (collision-safe), then
# stop. Useful to preview/curate what `make` would pick up.
flatten:
	bash "$(GATHER)" "$(CURDIR)" "$(SPOTS_DIR)"

check-fiji:
	@if [ -z "$(FIJI)" ] || [ ! -x "$(FIJI)" ]; then \
	  echo "ERROR: Fiji not found in any standard location."; \
	  echo ""; \
	  echo "Run 'make diagnose' to see what was checked, OR pass the exact"; \
	  echo "launcher path explicitly:"; \
	  echo ""; \
	  echo "  make FIJI=/Applications/Fiji.app/Contents/MacOS/<launcher-name>"; \
	  echo ""; \
	  echo "Or run 'make install-fiji' to download Fiji into $(FIJI_INSTALL_DIR)."; \
	  exit 1; \
	fi

install-fiji:
	@if [ -d "$(FIJI_INSTALL_DIR)/Fiji" ] || [ -d "$(FIJI_INSTALL_DIR)/Fiji.app" ]; then \
	  echo "Fiji already installed at $(FIJI_INSTALL_DIR)."; exit 0; \
	fi
	@mkdir -p "$(FIJI_INSTALL_DIR)"
	@echo "Downloading Fiji from $(FIJI_URL) ..."
	@echo "Installing into: $(FIJI_INSTALL_DIR)"
	curl -fL --progress-bar "$(FIJI_URL)" -o "$(FIJI_INSTALL_DIR)/fiji-macosx.zip"
	cd "$(FIJI_INSTALL_DIR)" && unzip -q fiji-macosx.zip && rm -f fiji-macosx.zip
	xattr -dr com.apple.quarantine "$(FIJI_INSTALL_DIR)/Fiji" 2>/dev/null || true
	chmod +x "$(FIJI_INSTALL_DIR)/Fiji/fiji" 2>/dev/null || true
	@echo "Installed: $(FIJI_INSTALL_DIR)/Fiji"

print:
	@echo "== Imaging profile =="
	@echo "  PROFILE:         $(PROFILE)   (phase | gfp)"
	@echo "  UM_PER_PIXEL:    $(UM_PER_PIXEL)"
	@echo "  FRAME_INTERVAL_S:$(FRAME_INTERVAL_S)"
	@echo "  LAG:             $(LAG)   frames = $$(( $(LAG) * $(FRAME_INTERVAL_S) )) s"
	@echo "== Fiji =="
	@echo "  FIJI:            $(if $(FIJI),$(FIJI),NOT FOUND - pass FIJI=... or run make install-fiji)"
	@echo "  PYTHON:          $(PYTHON)"
	@echo "== Stage 1: LoG detector =="
	@echo "  TARGET_CHANNEL:  $(TARGET_CHANNEL)"
	@echo "  DIAMETER:        $(DIAMETER)   (px; RADIUS = DIAMETER/2)"
	@echo "  THRESHOLD:       $(THRESHOLD)"
	@echo "  MEDIAN_FILTERING:$(MEDIAN_FILTERING)"
	@echo "  SUBPIXEL:        $(SUBPIXEL)"
	@echo "  QUALITY:         $(QUALITY)   (initial-spot filter)"
	@echo "== Stage 1: tracker ($(TRACKER)) =="
	@echo "  LINKING_MAX_DISTANCE:     $(LINKING_MAX_DISTANCE)"
	@echo "  GAP_CLOSING_MAX_DISTANCE: $(GAP_CLOSING_MAX_DISTANCE)"
	@echo "  MAX_FRAME_GAP:            $(MAX_FRAME_GAP)"
	@echo "  ALLOW_GAP_CLOSING:        $(ALLOW_GAP_CLOSING)   (lap only)"
	@echo "  ALLOW_SPLITTING:          $(ALLOW_SPLITTING)  (lap only)"
	@echo "  SPLITTING_MAX_DISTANCE:   $(SPLITTING_MAX_DISTANCE)  (lap only)"
	@echo "  ALLOW_MERGING:            $(ALLOW_MERGING)  (lap only)"
	@echo "  MERGING_MAX_DISTANCE:     $(MERGING_MAX_DISTANCE)  (lap only)"
	@echo "== Stages 2-3: MSD =="
	@echo "  LAG:             $(LAG)   (frames)"
	@echo "  MIN_CONSEC:      $(MIN_CONSEC)"
	@echo "  UM_PER_PIXEL:    $(UM_PER_PIXEL)"
	@echo "== Inputs =="
	@if [ -z "$(strip $(AVIS))" ]; then \
	  echo "  (no *.avi at top level; 'make' will gather from subfolders:)"; \
	  find . -type f -iname '*.avi' -not -iname '*_tracks.avi' \
	    -not -path './Fiji/*' -not -path './Fiji.app/*' \
	    -not -path './$(SPOTS_DIR)/*' -not -path '*/.*' | sed 's|^|    |' | sort; \
	else \
	  printf '  %s\n' $(AVIS); \
	fi

diagnose:
	@echo "== Fiji search =="
	@echo "Picked: $(if $(FIJI),$(FIJI),NONE)"
	@echo ""
	@echo "Candidates checked (* = exists):"
	@for p in $(FIJI_CANDIDATES); do \
	  if [ -e "$$p" ]; then echo "  * $$p"; else echo "    $$p"; fi; \
	done

clean:
	rm -rf "$(SPOTS_DIR)" "$(MSD_DIR)"
	@echo "Removed $(SPOTS_DIR)/ $(MSD_DIR)/"

help:
	@sed -n '1,/^SHELL/p' Makefile | sed '$$d' | sed 's/^# \{0,1\}//'
