import requests
import csv

scans_list_url = "http://spathi.cmpt.sfu.ca/multiscan/webui/scans/list"
output_path = "/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/real_scan_process/scans_status.csv"

# StorageFurniture,Table,TrashCan,Refrigerator,Oven,Microwave,WashingMachine,Dishwasher,Safe,Box,Suitcase,Unknown
# Not start,In progress,Done
# Shawn
# Bad,Okay,Good

def get_web_scans_list(scans_list_url):
    res = requests.get(scans_list_url)
    scans_list = res.json().get('data', None)
    return scans_list


if __name__ == '__main__':
    file = open(output_path, "w")
    csv_write = csv.writer(file)
    csv_head = ["Check Name", "Semantic Anno Name", "Arti Anno Name","Scan Index", "Object Index", "Scan Id", "Repeat", "Model Cat", "Model Quality", "Model Issues", "Semantic Annotation Status", "Semantic Anno Time", "Arti Anno Status", "Arti Anno Time", "Comments", "Model Link", "Semantic Anno Link", "Arti Anno Link", "Other Info"]
    csv_write.writerow(csv_head)

    # The scans list is in the scan time order (from early to late)
    scans_list = get_web_scans_list(scans_list_url)
    scan_index = 0
    scan_ids = []
    ikea_ids = []
    id_object_map = {}
    for scan in scans_list:
        if "sceneName" in scan and "motionnet" in scan["sceneName"]:
            # scan["objects"] is the order, like 19-60415054
            scan_index += 1
            if "objects" not in scan.keys():
                import pdb
                pdb.set_trace()
            repeat = "False"
            # Get the object index and the ikea id if it has one
            if '-' in scan["objects"]:
                split_index = scan["objects"].find('-')
                object_index = scan["objects"][:split_index]
                ikea_id = scan['objects'][split_index+1:]
                other_info = f"{scan['sceneName']}.{ikea_id}"
                if ikea_id in ikea_ids and not object_index == id_object_map[ikea_id]:
                    repeat = f"True {id_object_map[ikea_id]}"
                elif ikea_id not in ikea_ids:
                    ikea_ids.append(ikea_id)
                    id_object_map[ikea_id] = object_index
            else:
                object_index = scan["objects"]
                other_info = scan['sceneName']
            scan_id = scan["_id"]
            
            model_link = f"http://spathi.cmpt.sfu.ca/scene-toolkit/model-viewer?extra&modelId=multiscan.{scan_id}&format=textured-v1.1"
            semantic_anno_link = f"https://aspis.cmpt.sfu.ca/stk-multiscan/multiscan/segment-annotator?condition=manual&format=textured-v1.1&segmentType=triseg-finest-hier-v1.1&modelId=multiscan.{scan_id}&taskMode=fixup&startFrom=latest"
            arti_anno_link = f"https://aspis.cmpt.sfu.ca/stk-multiscan/motion-annotator?labelType=object-part&useDatGui=true&modelId=multiscan.{scan_id}"

            data = ["", "", "", scan_index, object_index, scan_id, repeat, "", "", "", "", "", "", "", "", model_link, semantic_anno_link, arti_anno_link, other_info]
            csv_write.writerow(data)
    
    file.close()