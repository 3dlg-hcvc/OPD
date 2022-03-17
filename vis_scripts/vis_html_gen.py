import json
import os
import bs4
import multiprocessing
from time import time
from alive_progress import alive_bar

ALLDATAPATH = "/project/3dlg-hcvc/motionnet/www/eccv-opdet-synthetic/render"
HTMLPATH = "/project/3dlg-hcvc/motionnet/www/eccv-opdet-synthetic/html"
VALIDLISTFILE = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/vis_scripts/valid_all.json"
HTMLTEMPLATE = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/vis_scripts/template.html"

HTMLURLBASE = "https://aspis.cmpt.sfu.ca/projects/motionnet/eccv-opdet-synthetic/"

# The sort method determines the order of the visualizatons in the html
# motion_f1 is for det+motion
SORT_METHOD = ['detect_f1', 'motion_f1']

MODELS = [
    "ocv0_rgbd",
    "cc_rgbd",
    "mf_rgbd",
    "rm_rgbd",
    "ancsh_rgb",
    "pc_rgb",
]
FINSIHED_MODELS = [
]

NUMEACHPAGE = 30
NUMPREDICTION = 5

CATMAP = {0: "drawer", 1: "door", 2: "lid"}
TYPEMAP = {0: "rotation", 1: "translation"}

SCORE_THRESHOLD = 0.8


def getImageTD(
    soup, img_path, background_path=None, width="256px", height="256px", lazy_load=True
):
    img_box = soup.new_tag("td")

    if not background_path == None:
        img_box.attrs.update(
            {
                "style": f"background:url('{background_path}') no-repeat",
            }
        )

    if lazy_load:
        img_tag = soup.new_tag("img", width=width, height=height, loading="lazy")
        img_tag.attrs.update({"data-src": img_path, "class": "lazyload"})
    else:
        img_tag = soup.new_tag("img", width=width, height=height, src=img_path)

    img_box.append(img_tag)
    return img_box


def getAnnoTD(soup, annotation, new_img_name):
    global CATMAP, TYPEMAP

    # Add corresponding evaluation info
    img_box = soup.new_tag("td")
    # Add the image name
    text_box = soup.new_tag("p")
    text_box.attrs.update({"style": "margin:0px"})
    img_box.append(text_box)
    text_box.string = f"{new_img_name}"
    # Add match information
    text_box = soup.new_tag("p")
    text_box.attrs.update({"style": "margin:0px"})
    img_box.append(text_box)
    eva_result = annotation["prediction"][new_img_name]
    if annotation["prediction"][new_img_name]["map_detect"] == -1:
        text_box.string = f"Not match in mAP case!!!"
    else:
        text_box.string = f"Match in mAP case"
    # Add the max_iou Info
    text_box = soup.new_tag("p")
    text_box.attrs.update({"style": "margin:0px"})
    img_box.append(text_box)
    text_box.string = f"Max IoU: {round(eva_result['iou'], 3)} with gt part {eva_result['gt']['partId']}"
    # Add the split
    text_box = soup.new_tag("p")
    text_box.attrs.update({"style": "margin:0px"})
    img_box.append(text_box)
    text_box.string = f" "
    # Add the predicted information
    text_box = soup.new_tag("p")
    text_box.attrs.update({"style": "margin:0px"})
    img_box.append(text_box)
    text_box.string = f"Pred: cat->{CATMAP[eva_result['pred']['category_id']]}; mtype->{TYPEMAP[eva_result['pred']['type']]}"
    # Add the gt information
    text_box = soup.new_tag("p")
    text_box.attrs.update({"style": "margin:0px"})
    img_box.append(text_box)
    text_box.string = (
        f"GT: cat->{eva_result['gt']['label']}; mtype->{eva_result['gt']['type']}"
    )
    # Add the split
    text_box = soup.new_tag("p")
    text_box.attrs.update({"style": "margin:0px"})
    img_box.append(text_box)
    text_box.string = f" "
    # Add the confidence score
    text_box = soup.new_tag("p")
    text_box.attrs.update({"style": "margin:0px"})
    img_box.append(text_box)
    text_box.string = f"Confidence: {round(eva_result['pred']['score'], 3)}"
    # Add axis error information
    text_box = soup.new_tag("p")
    text_box.attrs.update({"style": "margin:0px"})
    img_box.append(text_box)
    text_box.string = f"Axis Error: {round(eva_result['axis_error'], 3)}"
    # Add origin error information
    if (
        eva_result["gt"]["type"] == "rotation"
        or TYPEMAP[eva_result["pred"]["type"]] == "rotation"
    ):
        text_box = soup.new_tag("p")
        text_box.attrs.update({"style": "margin:0px"})
        img_box.append(text_box)
        text_box.string = f"Origin Error: {round(eva_result['origin_error'], 3)}"
    return img_box


def getPageChange(
    soup,
    page,
    page_num,
    prev_link,
    next_link,
    first_link=None,
    last_link=None,
    main_link=None,
):
    page_change = soup.new_tag("p")
    if not main_link == None:
        main_page = soup.new_tag("a")
        page_change.append(main_page)
        main_page.attrs.update({"href": main_link})
        main_page.string = "Main Page"
        page_change.append(" ")

    if not first_link == None:
        first_page = soup.new_tag("a")
        page_change.append(first_page)
        first_page.attrs.update({"href": first_link})
        first_page.string = "First Page"
        page_change.append(" ")

    if not page == 0:
        prev_page = soup.new_tag("a")
        page_change.append(prev_page)
        prev_page.attrs.update({"href": prev_link})
        prev_page.string = "Prev Page"
    page_change.append(f" {page+1}/{page_num} ")
    if not page == page_num - 1:
        next_page = soup.new_tag("a")
        page_change.append(next_page)
        next_page.attrs.update({"href": next_link})
        next_page.string = "Next Page"
        page_change.append(" ")

    if not last_link == None:
        last_page = soup.new_tag("a")
        page_change.append(last_page)
        last_page.attrs.update({"href": last_link})
        last_page.string = "Last Page"
        page_change.append(" ")
    return page_change


def createModelHTML(model, anno, sort_method):
    global HTMLPATH, HTMLTEMPLATE, HTMLURLBASE, NUMEACHPAGE, NUMPREDICTION, SCORE_THRESHOLD

    partIds = anno["gt"]
    annotations = anno[model]
    sort_list = sorted(
        annotations.items(), key=lambda i: i[1][sort_method], reverse=True
    )
    img_num = len(sort_list)
    page_num = img_num // NUMEACHPAGE + 1
    if img_num % NUMEACHPAGE == 0:
        page_num -= 1

    for page in range(page_num):
        template_file = open(HTMLTEMPLATE)
        template = template_file.read()
        soup = bs4.BeautifulSoup(template, features="html.parser")

        # Modify the tag title
        new_tag_title = soup.new_tag("title")
        soup.head.append(new_tag_title)
        new_tag_title.string = f"{model} Visualization (Sort based on {sort_method})"

        # Add the title for the page
        new_page_title = soup.new_tag("h1")
        soup.body.append(new_page_title)
        new_page_title.string = (
            f"{model} Page {page+1} (score_threhold {SCORE_THRESHOLD}) (Sort based on {sort_method})"
        )

        # Add the page link
        page_change = getPageChange(
            soup,
            page,
            page_num,
            prev_link=f"{HTMLURLBASE}/html/{model}_{sort_method}/{page-1}.html",
            next_link=f"{HTMLURLBASE}/html/{model}_{sort_method}/{page+1}.html",
            first_link=f"{HTMLURLBASE}/html/{model}_{sort_method}/{0}.html",
            last_link=f"{HTMLURLBASE}/html/{model}_{sort_method}/{page_num-1}.html",
            main_link=f"{HTMLURLBASE}/html/index.html",
        )
        soup.body.append(page_change)

        # Add the table
        table = soup.new_tag("table")
        soup.body.append(table)

        # Add the table title
        table_title = soup.new_tag("tr")
        table.append(table_title)
        title_box = soup.new_tag("th")
        table_title.append(title_box)
        title_box.string = "Image"
        for pred_index in range(NUMPREDICTION):
            title_box = soup.new_tag("th")
            table_title.append(title_box)
            title_box.string = f"Prediction {pred_index}"

        for img_index in range(NUMEACHPAGE):
            index = page * NUMEACHPAGE + img_index
            if index >= len(sort_list):
                break
            img_name = sort_list[index][0]

            # Add the GT visualization row
            image_div = soup.new_tag("tr")
            table.append(image_div)

            img_box = soup.new_tag("td")
            image_div.append(img_box)
            img_box.string = "GT"

            gt_part_num = len(partIds[img_name]["partIds"].keys())
            for gt_index in range(NUMPREDICTION):
                if gt_index >= gt_part_num:
                    img_box = soup.new_tag("td")
                    image_div.append(img_box)
                else:
                    partId = list(partIds[img_name]["partIds"].keys())[gt_index]
                    new_img_name = partIds[img_name]["partIds"][partId]
                    if model.split("_")[1] == "depth":
                        background_path = f"{HTMLURLBASE}/render/depth/{img_name}_d.png"
                    else:
                        background_path = f"{HTMLURLBASE}/render/valid/{img_name}.png"
                    image_box = getImageTD(
                        soup,
                        f"{HTMLURLBASE}/render/gt/{new_img_name}",
                        background_path=background_path,
                    )
                    image_div.append(image_box)

            # Add the image row based on the image index of the
            image_div = soup.new_tag("tr")
            table.append(image_div)

            img_box = soup.new_tag("td")
            image_div.append(img_box)
            text_box = soup.new_tag("p")
            text_box.attrs.update({"style": "margin:0px"})
            img_box.append(text_box)
            text_box.string = f"{img_name}.png"
            # Add the total_pred
            text_box = soup.new_tag("p")
            text_box.attrs.update({"style": "margin:0px"})
            img_box.append(text_box)
            text_box.string = f"num_pred: {sort_list[index][1]['total_pred']}"

            img_box.append(soup.new_tag("br"))

            # Add the overall detect precision, recall and f1
            text_box = soup.new_tag("p")
            text_box.attrs.update({"style": "margin:0px"})
            img_box.append(text_box)
            text_box.string = (
                f"detect_f1: {round(sort_list[index][1]['detect_f1'], 3)}"
            )
            text_box = soup.new_tag("p")
            text_box.attrs.update({"style": "margin:0px"})
            img_box.append(text_box)
            text_box.string = (
                f"detect_precision: {round(sort_list[index][1]['detect_precision'], 3)}"
            )
            text_box = soup.new_tag("p")
            text_box.attrs.update({"style": "margin:0px"})
            img_box.append(text_box)
            text_box.string = (
                f"detect_recall: {round(sort_list[index][1]['detect_recall'], 3)}"
            )
            # Add mAP_detection
            text_box = soup.new_tag("p")
            text_box.attrs.update({"style": "margin:0px"})
            img_box.append(text_box)
            text_box.string = (
                f"mAP_detection: {round(sort_list[index][1]['map_detect'], 3)}"
            )

            img_box.append(soup.new_tag("br"))

            # Add the overall detect+motion precision, recall and f1
            text_box = soup.new_tag("p")
            text_box.attrs.update({"style": "margin:0px"})
            img_box.append(text_box)
            text_box.string = (
                f"motion_f1: {round(sort_list[index][1]['motion_f1'], 3)}"
            )
            text_box = soup.new_tag("p")
            text_box.attrs.update({"style": "margin:0px"})
            img_box.append(text_box)
            text_box.string = (
                f"motion_precision: {round(sort_list[index][1]['motion_precision'], 3)}"
            )
            text_box = soup.new_tag("p")
            text_box.attrs.update({"style": "margin:0px"})
            img_box.append(text_box)
            text_box.string = (
                f"motion_recall: {round(sort_list[index][1]['motion_recall'], 3)}"
            )

            # Add type precision
            text_box = soup.new_tag("p")
            text_box.attrs.update({"style": "margin:0px"})
            img_box.append(text_box)
            text_box.string = (
                f"type_precision: {round(sort_list[index][1]['type_precision'], 3)}"
            )

            # Add annotations row
            anno_div = soup.new_tag("tr")
            table.append(anno_div)

            img_box = soup.new_tag("td")
            anno_div.append(img_box)
            img_box.string = "Details for Above"

            # Add the predicted images
            pred_num = len(sort_list[index][1]["prediction"].keys())
            matched_pred = {}
            unmatched_pred = []

            for pred_index in range(pred_num):
                new_img_name = f"{img_name}__{pred_index}.png"
                eva_result = sort_list[index][1]["prediction"][new_img_name]
                if eva_result["map_detect"] == -1:
                    unmatched_pred.append(pred_index)
                else:
                    matched_pred[eva_result["gt"]["partId"]] = new_img_name

            for vis_index in range(NUMPREDICTION):
                if vis_index < gt_part_num:
                    partId = list(partIds[img_name]["partIds"].keys())[vis_index]
                    if partId in matched_pred.keys():
                        # Add corresponding image
                        new_img_name = matched_pred[partId]
                        if model.split("_")[1] == "depth":
                            background_path = (
                                f"{HTMLURLBASE}/render/depth/{img_name}_d.png"
                            )
                        else:
                            background_path = (
                                f"{HTMLURLBASE}/render/valid/{img_name}.png"
                            )
                        image_box = getImageTD(
                            soup,
                            f"{HTMLURLBASE}/render/{model}/{new_img_name}",
                            background_path=background_path,
                        )
                        image_div.append(image_box)
                        # Add corresponding evaluation info
                        img_box = getAnnoTD(soup, sort_list[index][1], new_img_name)
                        anno_div.append(img_box)
                    else:
                        img_box = soup.new_tag("td")
                        image_div.append(img_box)
                        img_box = soup.new_tag("td")
                        anno_div.append(img_box)
                else:
                    if vis_index >= gt_part_num + len(unmatched_pred):
                        img_box = soup.new_tag("td")
                        image_div.append(img_box)
                        img_box = soup.new_tag("td")
                        anno_div.append(img_box)
                    else:
                        unmatched_index = vis_index - gt_part_num
                        new_img_name = (
                            f"{img_name}__{unmatched_pred[unmatched_index]}.png"
                        )
                        if model.split("_")[1] == "depth":
                            background_path = (
                                f"{HTMLURLBASE}/render/depth/{img_name}_d.png"
                            )
                        else:
                            background_path = (
                                f"{HTMLURLBASE}/render/valid/{img_name}.png"
                            )
                        image_box = getImageTD(
                            soup,
                            f"{HTMLURLBASE}/render/{model}/{new_img_name}",
                            background_path=background_path,
                        )
                        image_div.append(image_box)
                        # Add corresponding evaluation info
                        img_box = getAnnoTD(soup, sort_list[index][1], new_img_name)
                        anno_div.append(img_box)

        # Add the page link
        page_change = getPageChange(
            soup,
            page,
            page_num,
            prev_link=f"{HTMLURLBASE}/html/{model}_{sort_method}/{page-1}.html",
            next_link=f"{HTMLURLBASE}/html/{model}_{sort_method}/{page+1}.html",
            first_link=f"{HTMLURLBASE}/html/{model}_{sort_method}/{0}.html",
            last_link=f"{HTMLURLBASE}/html/{model}_{sort_method}/{page_num-1}.html",
            main_link=f"{HTMLURLBASE}/html/index.html",
        )
        soup.body.append(page_change)

        page_path = f"{HTMLPATH}/{model}_{sort_method}/{page}.html"
        page_html = open(page_path, "w")
        page_html.write(str(soup))


def createComparisonHTML(anno):
    global HTMLPATH, HTMLTEMPLATE, HTMLURLBASE, MODELS, NUMPREDICTION, SCORE_THRESHOLD

    partIds = anno["gt"]
    # Sorted based on -O RGBD
    annotations = anno["ocv0_rgbd"]
    sort_list = sorted(
        annotations.items(), key=lambda i: i[1]["motion_f1"], reverse=True
    )
    # image_names = list(anno["gt"].keys())
    # Visualize one image in one page
    page_num = len(sort_list)

    for page in range(page_num):
        template_file = open(HTMLTEMPLATE)
        template = template_file.read()
        soup = bs4.BeautifulSoup(template, features="html.parser")

        # Modify the tag title
        new_tag_title = soup.new_tag("title")
        soup.head.append(new_tag_title)
        new_tag_title.string = f"Comparison among Models"

        # Add the title for the page
        new_page_title = soup.new_tag("h1")
        soup.body.append(new_page_title)
        new_page_title.string = (
            f"Comparison Page {page+1} (score_threhold {SCORE_THRESHOLD})"
        )

        # Add the page link
        page_change = getPageChange(
            soup,
            page,
            page_num,
            prev_link=f"{HTMLURLBASE}/html/comparison/{page-1}.html",
            next_link=f"{HTMLURLBASE}/html/comparison/{page+1}.html",
            first_link=f"{HTMLURLBASE}/html/comparison/{0}.html",
            last_link=f"{HTMLURLBASE}/html/comparison/{page_num-1}.html",
            main_link=f"{HTMLURLBASE}/html/index.html",
        )
        soup.body.append(page_change)

        # Add the table
        table = soup.new_tag("table")
        soup.body.append(table)

        # Add the table title
        table_title = soup.new_tag("tr")
        table.append(table_title)
        title_box = soup.new_tag("th")
        table_title.append(title_box)
        title_box.string = f"Model for {sort_list[page][0]}"
        for pred_index in range(NUMPREDICTION):
            title_box = soup.new_tag("th")
            table_title.append(title_box)
            title_box.string = f"Image {pred_index}"

        # Only put one image in one page
        index = page
        img_name = sort_list[index][0]

        # Add the GT visualization row
        image_div = soup.new_tag("tr")
        table.append(image_div)

        img_box = soup.new_tag("td")
        image_div.append(img_box)
        img_box.string = "GT"

        gt_part_num = len(partIds[img_name]["partIds"].keys())
        for gt_index in range(NUMPREDICTION):
            if gt_index >= gt_part_num:
                img_box = soup.new_tag("td")
                image_div.append(img_box)
            else:
                partId = list(partIds[img_name]["partIds"].keys())[gt_index]
                new_img_name = partIds[img_name]["partIds"][partId]
                background_path = f"{HTMLURLBASE}/render/valid/{img_name}.png"
                image_box = getImageTD(
                    soup,
                    f"{HTMLURLBASE}/render/gt/{new_img_name}",
                    background_path=background_path,
                )
                image_div.append(image_box)

        for model in MODELS:
            # Add the image row based on the image index of the
            image_div = soup.new_tag("tr")
            table.append(image_div)

            img_box = soup.new_tag("td")
            image_div.append(img_box)
            text_box = soup.new_tag("p")
            img_box.append(text_box)
            text_box.string = f"{model}"
            text_box = soup.new_tag("p")
            img_box.append(text_box)
            text_box.string = f"{img_name}.png"
            # Add the total_pred
            text_box = soup.new_tag("p")
            img_box.append(text_box)
            text_box.string = f"num_pred: {anno[model][img_name]['total_pred']}"
            # Add the detect_precison
            text_box = soup.new_tag("p")
            img_box.append(text_box)
            text_box.string = f"detect_f1: {round(anno[model][img_name]['detect_f1'], 3)}"
            text_box = soup.new_tag("p")
            img_box.append(text_box)
            text_box.string = f"detect_precision: {round(anno[model][img_name]['detect_precision'], 3)}"
            text_box = soup.new_tag("p")
            img_box.append(text_box)
            text_box.string = f"detect_recall: {round(anno[model][img_name]['detect_recall'], 3)}"
            # Add mAP_detection
            text_box = soup.new_tag("p")
            img_box.append(text_box)
            text_box.string = (
                f"mAP_detection: {round(anno[model][img_name]['map_detect'], 3)}"
            )
            # Add type precision
            text_box = soup.new_tag("p")
            img_box.append(text_box)
            text_box.string = (
                f"type_precision: {round(anno[model][img_name]['type_precision'], 3)}"
            )

            # Add annotations row
            anno_div = soup.new_tag("tr")
            table.append(anno_div)

            img_box = soup.new_tag("td")
            anno_div.append(img_box)
            img_box.string = "Details for Above"

            # Add the predicted images
            pred_num = len(anno[model][img_name]["prediction"].keys())
            matched_pred = {}
            unmatched_pred = []

            for pred_index in range(pred_num):
                new_img_name = f"{img_name}__{pred_index}.png"
                eva_result = anno[model][img_name]["prediction"][new_img_name]
                if eva_result["map_detect"] == -1:
                    unmatched_pred.append(pred_index)
                else:
                    matched_pred[eva_result["gt"]["partId"]] = new_img_name

            for vis_index in range(NUMPREDICTION):
                if vis_index < gt_part_num:
                    partId = list(partIds[img_name]["partIds"].keys())[vis_index]
                    if partId in matched_pred.keys():
                        # Add corresponding image
                        new_img_name = matched_pred[partId]
                        if model.split("_")[1] == "depth":
                            background_path = (
                                f"{HTMLURLBASE}/render/depth/{img_name}_d.png"
                            )
                        else:
                            background_path = (
                                f"{HTMLURLBASE}/render/valid/{img_name}.png"
                            )
                        image_box = getImageTD(
                            soup,
                            f"{HTMLURLBASE}/render/{model}/{new_img_name}",
                            background_path=background_path,
                        )
                        image_div.append(image_box)
                        # Add corresponding evaluation info
                        img_box = getAnnoTD(soup, anno[model][img_name], new_img_name)
                        anno_div.append(img_box)
                    else:
                        img_box = soup.new_tag("td")
                        image_div.append(img_box)
                        img_box = soup.new_tag("td")
                        anno_div.append(img_box)
                else:
                    if vis_index >= gt_part_num + len(unmatched_pred):
                        img_box = soup.new_tag("td")
                        image_div.append(img_box)
                        img_box = soup.new_tag("td")
                        anno_div.append(img_box)
                    else:
                        unmatched_index = vis_index - gt_part_num
                        new_img_name = (
                            f"{img_name}__{unmatched_pred[unmatched_index]}.png"
                        )
                        if model.split("_")[1] == "depth":
                            background_path = (
                                f"{HTMLURLBASE}/render/depth/{img_name}_d.png"
                            )
                        else:
                            background_path = (
                                f"{HTMLURLBASE}/render/valid/{img_name}.png"
                            )
                        image_box = getImageTD(
                            soup,
                            f"{HTMLURLBASE}/render/{model}/{new_img_name}",
                            background_path=background_path,
                        )
                        image_div.append(image_box)
                        # Add corresponding evaluation info
                        img_box = getAnnoTD(soup, anno[model][img_name], new_img_name)
                        anno_div.append(img_box)

        # Add the page link
        page_change = getPageChange(
            soup,
            page,
            page_num,
            prev_link=f"{HTMLURLBASE}/html/comparison/{page-1}.html",
            next_link=f"{HTMLURLBASE}/html/comparison/{page+1}.html",
            first_link=f"{HTMLURLBASE}/html/comparison/{0}.html",
            last_link=f"{HTMLURLBASE}/html/comparison/{page_num-1}.html",
            main_link=f"{HTMLURLBASE}/html/index.html",
        )
        soup.body.append(page_change)

        page_path = f"{HTMLPATH}/comparison/{page}.html"
        page_html = open(page_path, "w")
        page_html.write(str(soup))


def createMainHTML():
    global SORT_METHOD, HTMLPATH, HTMLTEMPLATE, HTMLURLBASE

    template_file = open(HTMLTEMPLATE)
    template = template_file.read()
    soup = bs4.BeautifulSoup(template, features="html.parser")

    # Modify the tag title
    new_tag_title = soup.new_tag("title")
    soup.head.append(new_tag_title)
    new_tag_title.string = f"OPDet Visualization"

    # Add the title for the page
    new_page_title = soup.new_tag("h1")
    soup.body.append(new_page_title)
    new_page_title.string = f"Main Page for OPDet Visualization"

    # Add the comparison page
    main_page = soup.new_tag("a")
    soup.body.append(main_page)
    main_page.attrs.update({"href": f"{HTMLURLBASE}/html/comparison/0.html"})
    main_page.string = (
        "Comparison Page (Compare the visualizations of different models)"
    )

    soup.body.append(soup.new_tag("br"))

    for sort_method in SORT_METHOD:
        # Add the title for the sort method
        new_page_title = soup.new_tag("h2")
        soup.body.append(new_page_title)
        new_page_title.string = f"Sort based on {sort_method}"

        for model in MODELS:
            # Add the page for different MODELS
            main_page = soup.new_tag("a")
            soup.body.append(main_page)
            main_page.attrs.update({"href": f"{HTMLURLBASE}/html/{model}_{sort_method}/0.html"})
            main_page.string = f"Visualization Page for {model}"
            soup.body.append(soup.new_tag("br"))

        page_path = f"{HTMLPATH}/index.html"
        page_html = open(page_path, "w")
        page_html.write(str(soup))


if __name__ == "__main__":
    start = time()
    pool = multiprocessing.Pool(processes=16)

    # Load relevant data
    valid_image_file = open(VALIDLISTFILE)
    selection = json.load(valid_image_file)
    valid_image_file.close()

    anno = {}
    # Load annotations for gt
    anno["gt"] = {}
    annotation_file = open(f"{ALLDATAPATH}/annotations/instance_render_gt.json")
    annotation = json.load(annotation_file)
    annotation_file.close()

    print("Load GT Annotation")
    with alive_bar(len(selection)) as bar:
        for select_img in selection:
            anno["gt"][select_img] = {"partIds": {}}
            for img_name in annotation.keys():
                if img_name.split("__")[0] == select_img:
                    anno["gt"][select_img]["partIds"][
                        annotation[img_name]["partId"]
                    ] = img_name
            bar()

    # Load annotations for each model
    print("Loading Model Annotation")
    with alive_bar(len(MODELS) * len(selection)) as bar:
        for model in MODELS:
            anno[model] = {}
            # Read the raw annotation file
            annotation_file = open(
                f"{ALLDATAPATH}/annotations/instance_render_{model}.json"
            )
            annotation = json.load(annotation_file)
            annotation_file.close()

            for select_img in selection:
                anno[model][select_img] = {
                    "map_detect": 0,
                    "type_precision": 0,
                    "detect_f1": 0,
                    "detect_precision": 0,
                    "detect_recall": 0,
                    "motion_f1": 0,
                    "motion_precision": 0,
                    "motion_recall": 0,
                    "total_pred": 0,
                    "prediction": {},
                }
                for img_name in annotation.keys():
                    if img_name.split("__")[0] == select_img:
                        # If the prediction is the first one, then use the map_detect and type_precision
                        if (img_name.split("__")[1]).split(".")[0] == "0":
                            if annotation[img_name]["map_detect"] != -1:
                                anno[model][select_img]["map_detect"] = annotation[
                                    img_name
                                ]["map_detect"]
                            anno[model][select_img]["type_precision"] = annotation[
                                img_name
                            ]["type_precision"]
                            anno[model][select_img]["detect_f1"] = annotation[
                                img_name
                            ]["detect_f1"]
                            anno[model][select_img]["detect_precision"] = annotation[
                                img_name
                            ]["detect_precision"]
                            anno[model][select_img]["detect_recall"] = annotation[
                                img_name
                            ]["detect_recall"]
                            anno[model][select_img]["motion_f1"] = annotation[
                                img_name
                            ]["motion_f1"]
                            anno[model][select_img]["motion_precision"] = annotation[
                                img_name
                            ]["motion_precision"]
                            anno[model][select_img]["motion_recall"] = annotation[
                                img_name
                            ]["motion_recall"]
                            anno[model][select_img]["total_pred"] = annotation[img_name][
                                "total_pred"
                            ]
                        anno[model][select_img]["prediction"][img_name] = annotation[
                            img_name
                        ]
                bar()

    print("Generating the model HTMLs")
    with alive_bar(len(SORT_METHOD) * len(MODELS)) as bar:
        for sort_method in SORT_METHOD:
            os.makedirs(f"{HTMLPATH}", exist_ok=True)
            # Create html for each model
            for model in MODELS:
                if model in FINSIHED_MODELS:
                    continue
                os.makedirs(f"{HTMLPATH}/{model}_{sort_method}", exist_ok=True)
                pool.apply_async(createModelHTML, (model, anno, sort_method,))
                # createModelHTML(model, anno, sort_method)
                bar()

    pool.close()
    pool.join()

    print("Generating the comparison HTMLs")
    # Create html for comparison with different models
    os.makedirs(f"{HTMLPATH}/comparison", exist_ok=True)
    createComparisonHTML(anno)

    # Create the main html to connect all pages
    createMainHTML()

    stop = time()
    print(str(stop - start) + " seconds")
