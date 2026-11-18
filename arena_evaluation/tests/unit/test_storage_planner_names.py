"""Unit tests for arena_evaluation.storage.planner_names.split_planner_name.

Splits "local/inter" planner strings:

  "teb"                  -> ("teb", "none")
  "teb-dwa"              -> ("teb", "dwa")
  "a-local-inter"        -> ("local", "inter")     # first segment dropped
  "a-local-inter-extra"  -> ("local", "inter-extra")
"""

from __future__ import annotations




# falsy inputs


# one part


# two parts


# three or more parts: first segment dropped, remainder joined


# type contract


# hypothesis properties


