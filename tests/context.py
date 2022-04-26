import os
import sys

# Adds "forest_cover_type" to sys.path
# Now you can do import with "from forest_cover_type.Sub-Package ..."
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "forest_cover_type")),
)
