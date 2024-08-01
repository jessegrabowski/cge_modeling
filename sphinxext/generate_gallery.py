"""
Sphinx plugin to run generate a gallery for notebooks

Modified from the pymc project, whihch modified the seaborn project, which modified the mpld3 project.
"""

import base64
import json
import os
import shutil

from glob import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sphinx

from matplotlib import image

logger = sphinx.util.logging.getLogger(__name__)

DOC_SRC = os.path.dirname(os.path.abspath(__file__))

DEFAULT_IMG_LOC = None
external_nbs = {}

HEAD = """
Example Gallery
===============

.. toctree::
   :hidden:

"""

SECTION_TEMPLATE = """
.. _{section_id}:

{section_title}
{underlines}

.. grid:: 1 2 3 3
   :gutter: 4

"""

ITEM_TEMPLATE = """
   .. grid-item-card:: :doc:`{doc_name}`
      :img-top: {image}
      :link: {doc_reference}
      :link-type: {link_type}
      :shadow: none
"""

folder_title_map = {"getting_started": "Getting Started"}


def create_thumbnail(infile, width=275, height=275, cx=0.5, cy=0.5, border=4):
    """Overwrites `infile` with a new file of the given size"""
    im = image.imread(infile)
    rows, cols = im.shape[:2]
    size = min(rows, cols)
    if size == cols:
        xslice = slice(0, size)
        ymin = min(max(0, int(cx * rows - size // 2)), rows - size)
        yslice = slice(ymin, ymin + size)
    else:
        yslice = slice(0, size)
        xmin = min(max(0, int(cx * cols - size // 2)), cols - size)
        xslice = slice(xmin, xmin + size)
    thumb = im[yslice, xslice]
    thumb[:border, :, :3] = thumb[-border:, :, :3] = 0
    thumb[:, :border, :3] = thumb[:, -border:, :3] = 0

    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

    ax = fig.add_axes([0, 0, 1, 1], aspect="auto", frameon=False, xticks=[], yticks=[])
    ax.imshow(thumb, aspect="auto", resample=True, interpolation="bilinear")
    fig.savefig(infile, dpi=dpi)
    plt.close(fig)
    return fig


class NotebookGenerator:
    """Tools for generating an example page from a file"""

    def __init__(self, filename, root_dir, folder):
        self.folder = folder
        self.basename = os.path.basename(filename)
        self.stripped_name = os.path.splitext(self.basename)[0]
        self.image_dir = os.path.join(root_dir, "_thumbnails", folder)
        self.png_path = os.path.join(self.image_dir, f"{self.stripped_name}.png")
        with open(filename, encoding="utf-8") as fid:
            self.json_source = json.load(fid)
        self.default_image_loc = DEFAULT_IMG_LOC

    def extract_preview_pic(self):
        """By default, just uses the last image in the notebook."""
        pic = None
        for cell in self.json_source["cells"]:
            for output in cell.get("outputs", []):
                if "image/png" in output.get("data", []):
                    pic = output["data"]["image/png"]
        if pic is not None:
            return base64.b64decode(pic)
        return None

    def gen_previews(self):
        preview = self.extract_preview_pic()
        if preview is not None:
            print(self.png_path)
            with open(self.png_path, "wb") as buff:
                buff.write(preview)
        else:
            logger.warning(
                f"Didn't find any pictures in {self.basename}",
                type="thumbnail_extractor",
            )
            shutil.copy(self.default_image_loc, self.png_path)
        create_thumbnail(self.png_path)


def main(app):
    logger.info("Starting thumbnail extractor.")

    working_dir = os.getcwd()
    os.chdir(app.builder.srcdir)

    file = [HEAD]

    for folder, title in folder_title_map.items():
        file.append(
            SECTION_TEMPLATE.format(
                section_title=title, section_id=folder, underlines="-" * len(title)
            )
        )

        thumbnail_dir = os.path.join("..", "..", "_thumbnails", folder)
        if not os.path.exists(thumbnail_dir):
            os.makedirs(thumbnail_dir)

        if folder in external_nbs.keys():
            for descr in external_nbs[folder]:
                file.append(
                    ITEM_TEMPLATE.format(
                        doc_name=descr["doc_name"],
                        image=descr["image"],
                        doc_reference=descr["doc_reference"],
                        link_type=descr["link_type"],
                    )
                )

        nb_paths = glob(f"examples/{folder}/*.ipynb")
        for nb_path in nb_paths:
            nbg = NotebookGenerator(
                filename=nb_path, root_dir=os.path.join("..", ".."), folder=folder
            )
            nbg.gen_previews()

            file.append(
                ITEM_TEMPLATE.format(
                    doc_name=os.path.join(folder, nbg.stripped_name),
                    image="/" + nbg.png_path,
                    doc_reference=os.path.join(folder, nbg.stripped_name),
                    link_type="doc",
                )
            )

    with open(os.path.join("examples", "gallery.rst"), "w", encoding="utf-8") as f:
        f.write("\n".join(file))

    os.chdir(working_dir)


def setup(app):
    app.connect("builder-inited", main)
