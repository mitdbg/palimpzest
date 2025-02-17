"""

An illustrataive example of how predictors can help each other.

@kylel

"""

import json
import os
import pathlib

from papermage.magelib import Box, Document, Entity, Metadata, Span
from papermage.recipes import CoreRecipe
from papermage.visualizers.visualizer import plot_entities_on_page


def fix_layout_preds_with_tokens(doc, page_id=1):
    # layoutparser doesnt produce very tight boundaries
    before = plot_entities_on_page(
        page_image=doc.images[page_id], entities=[block for block in doc.layout if block.boxes[0].page == page_id]
    )

    # let's use tokens to constrain the layout
    blocks = []
    for block in doc.layout:
        block.tokens = doc.find_by_box(query=Entity(boxes=[block.boxes[0]]), field_name="tokens")
        token_boxes = [box for t in block.tokens for box in t.boxes]
        new_block_box = Box.create_enclosing_box(boxes=token_boxes)
        new_block = Entity(boxes=[new_block_box], metadata=block.metadata)
        blocks.append(new_block)

    # fixed!
    after = plot_entities_on_page(
        page_image=doc.images[page_id], entities=[block for block in blocks if block.boxes[0].page == page_id]
    )
    return before, after


# get doc
recipe = CoreRecipe()
fixture_path = pathlib.Path(__file__).parent / "tests/fixtures"
doc = recipe.from_pdf(pdf=fixture_path / "2304.02623v1.pdf")

# analyze
before, after = fix_layout_preds_with_tokens(doc=doc, page_id=1)
