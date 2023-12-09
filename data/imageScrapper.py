from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import os, requests, lxml, re, json, urllib.request
from bs4 import BeautifulSoup
from serpapi import GoogleSearch

# headers = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36"
# }
#
# params = {
#     "q": "mincraft wallpaper 4k",  # search query
#     "tbm": "isch",  # image results
#     "hl": "en",  # language of the search
#     "gl": "us",  # country where search comes from
#     "ijn": "0"  # page number
# }
#
# html = requests.get("https://www.google.com/search", params=params, headers=headers, timeout=30)
# soup = BeautifulSoup(html.text, "lxml")


def get_images_with_request_headers():
    del params["ijn"]
    params["content-type"] = "image/png"  # parameter that indicate the original media type

    return [img["src"] for img in soup.select("img")]


def get_suggested_search_data():
    suggested_searches = []

    all_script_tags = soup.select("script")

    # https://regex101.com/r/48UZhY/6
    matched_images = "".join(re.findall(r"AF_initDataCallback\(({key: 'ds:1'.*?)\);</script>", str(all_script_tags)))

    # https://kodlogs.com/34776/json-decoder-jsondecodeerror-expecting-property-name-enclosed-in-double-quotes
    # if you try to json.loads() without json.dumps it will throw an error:
    # "Expecting property name enclosed in double quotes"
    matched_images_data_fix = json.dumps(matched_images)
    matched_images_data_json = json.loads(matched_images_data_fix)

    # search for only suggested search thumbnails related
    # https://regex101.com/r/ITluak/2
    suggested_search_thumbnails = ",".join(re.findall(r'{key(.*?)\[null,\"Size\"', matched_images_data_json))

    # https://regex101.com/r/MyNLUk/1
    suggested_search_thumbnail_encoded = re.findall(r'\"(https:\/\/encrypted.*?)\"', suggested_search_thumbnails)

    for suggested_search, suggested_search_fixed_thumbnail in zip(soup.select(".PKhmud.sc-it.tzVsfd"),
                                                                  suggested_search_thumbnail_encoded):
        suggested_searches.append({
            "name": suggested_search.select_one(".VlHyHc").text,
            "link": f"https://www.google.com{suggested_search.a['href']}",
            # https://regex101.com/r/y51ZoC/1
            "chips": "".join(re.findall(r"&chips=(.*?)&", suggested_search.a["href"])),
            # https://stackoverflow.com/a/4004439/15164646 comment by Frédéric Hamidi
            "thumbnail": bytes(suggested_search_fixed_thumbnail, "ascii").decode("unicode-escape")
        })

    return suggested_searches


def get_original_images():
    """
    https://kodlogs.com/34776/json-decoder-jsondecodeerror-expecting-property-name-enclosed-in-double-quotes
    if you try to json.loads() without json.dumps() it will throw an error:
    "Expecting property name enclosed in double quotes"
    """

    google_images = []

    all_script_tags = soup.select("script")

    # # https://regex101.com/r/48UZhY/4
    matched_images_data = "".join(re.findall(r"AF_initDataCallback\(([^<]+)\);", str(all_script_tags)))

    matched_images_data_fix = json.dumps(matched_images_data)
    matched_images_data_json = json.loads(matched_images_data_fix)

    # https://regex101.com/r/VPz7f2/1
    matched_google_image_data = re.findall(r'\"b-GRID_STATE0\"(.*)sideChannel:\s?{}}', matched_images_data_json)

    # https://regex101.com/r/NnRg27/1
    matched_google_images_thumbnails = ", ".join(
        re.findall(r'\[\"(https\:\/\/encrypted-tbn0\.gstatic\.com\/images\?.*?)\",\d+,\d+\]',
                   str(matched_google_image_data))).split(", ")

    thumbnails = [
        bytes(bytes(thumbnail, "ascii").decode("unicode-escape"), "ascii").decode("unicode-escape") for thumbnail in
        matched_google_images_thumbnails
    ]

    # removing previously matched thumbnails for easier full resolution image matches.
    removed_matched_google_images_thumbnails = re.sub(
        r'\[\"(https\:\/\/encrypted-tbn0\.gstatic\.com\/images\?.*?)\",\d+,\d+\]', "", str(matched_google_image_data))

    # https://regex101.com/r/fXjfb1/4
    # https://stackoverflow.com/a/19821774/15164646
    matched_google_full_resolution_images = re.findall(r"(?:'|,),\[\"(https:|http.*?)\",\d+,\d+\]",
                                                       removed_matched_google_images_thumbnails)

    full_res_images = [
        bytes(bytes(img, "ascii").decode("unicode-escape"), "ascii").decode("unicode-escape") for img in
        matched_google_full_resolution_images
    ]

    for index, (metadata, thumbnail, original) in enumerate(
            zip(soup.select('.isv-r.PNCib.MSM1fd.BUooTd'), thumbnails, full_res_images), start=1):
        google_images.append({
            "title": metadata.select_one(".VFACy.kGQAp.sMi44c.lNHeqe.WGvvNb")["title"],
            "link": metadata.select_one(".VFACy.kGQAp.sMi44c.lNHeqe.WGvvNb")["href"],
            "source": metadata.select_one(".fxgdke").text,
            "thumbnail": thumbnail,
            "original": original
        })

        # Download original images
        print(f'Downloading {index} image...')

        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent',
                              'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36')]
        urllib.request.install_opener(opener)

        urllib.request.urlretrieve(original, f'Bs4_Images/original_size_img_{index}.jpg')

    return google_images


def serpapi_get_google_images():
    image_results = []

    for query in ["Coffee", "boat", "skyrim", "minecraft"]:
        # search query parameters
        params = {
            "engine": "google",  # search engine. Google, Bing, Yahoo, Naver, Baidu...
            "q": query,  # search query
            "tbm": "isch",  # image results
            "num": "100",  # number of images per page
            "ijn": 0,  # page number: 0 -> first page, 1 -> second...
            "api_key": os.getenv("API_KEY")  # your serpapi api key
            # other query parameters: hl (lang), gl (country), etc
        }

        search = GoogleSearch(params)  # where data extraction happens

        images_is_present = True
        while images_is_present:
            results = search.get_dict()  # JSON -> Python dictionary

            # checks for "Google hasn't returned any results for this query."
            if "error" not in results:
                for image in results["images_results"]:
                    if image["original"] not in image_results:
                        image_results.append(image["original"])

                # update to the next page
                params["ijn"] += 1
            else:
                images_is_present = False
                print(results["error"])

    # -----------------------
    # Downloading images

    for index, image in enumerate(results["images_results"], start=1):
        print(f"Downloading {index} image...")

        opener = urllib.request.build_opener()
        opener.addheaders = [("User-Agent",
                              "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36")]
        urllib.request.install_opener(opener)

        urllib.request.urlretrieve(image["original"], f"SerpApi_Images/original_size_img_{index}.jpg")

    print(json.dumps(image_results, indent=2))
    print(len(image_results))

if __name__ == '__main__':
    # driver = webdriver.Chrome('D:/chromedriver.exe')
    # driver.get('https://www.google.ca/imghp?hl=en&tab=ri&authuser=0&ogbl/')
    # # unsplashed.com
    # box = driver.find_element_by_xpath('/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input')
    # context = 'empty lecture hall'
    # box.send_keys(context)
    # box.send_keys(Keys.ENTER)
    # for i in range(1, 101):
    #     try:
    #         //*[@id="app"]/div/div[2]/div[4]/div[1]/div/div/div/div[2]/figure[1]/div/div/div/div/a/div/div[2]/div/img
    #         folder_path = './saved_pics/' + context + '/(' + context + str(i) + ').png'
    #         driver.find_element_by_xpath('//*[@id="islrg"]/div[1]/div[' + str(i) + ']/a[1]/div[1]/img').screenshot(
    #             './saved_pics/' + context + '(' + str(i) + ').png')  # get big image instead
    #     except:
    #         print("here")
    #         pass
    # driver.quit()

    params = {
        "api_key": "175b0bc6895ea2e5ebf74041bdaaed44db46b5326b7f81855fc9cd3b33797c95",
        "device": "desktop",
        "engine": "google",
        "q": "propose decoration",
        "location": "Austin, Texas, United States",
        "google_domain": "google.com",
        "gl": "us",
        "hl": "en",
        "tbm": "isch"
    }

    search = GoogleSearch(params)
    savePath = 'D:/Tianma/dataset/background/Positive/propose_decoration/'
    ID = 0
    for image_result in search.get_dict()['images_results']:
        link = image_result["original"]
        try:
            print("link: " + link)
            # wget.download(link, '.')
            img_data = requests.get(link).content
            with open(savePath + str(ID) + '.jpg', 'wb') as handler:
                handler.write(img_data)
                ID = ID + 1
        except:
            pass