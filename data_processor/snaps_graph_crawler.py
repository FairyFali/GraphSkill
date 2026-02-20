import requests
from bs4 import BeautifulSoup
import os
import json

# URL of the SNAP datasets page
url = 'https://snap.stanford.edu/data/index.html'

# Send a GET request to the URL
response = requests.get(url)
response.raise_for_status()  # Ensure the request was successful

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Dictionary to store datasets by type
datasets = {
    'directed': [],
    'undirected': [],
    'weighted&directed': [],
    'weighted&undirected': [],
    'others': []
}

def get_page_content(href):
    page_url = f'https://snap.stanford.edu/data/{href}'
    page_response = requests.get(page_url)
    page_response.raise_for_status()  # Ensure the request was successful
    page_soup = BeautifulSoup(page_response.text, 'html.parser')
    return page_soup

def get_download_href(page_soup):
    dataset_href = []
    for table in page_soup.find_all('table'):
        for row in table.find_all('tr')[1:]:
            cols = row.find_all('td')
            if len(cols) >= 2:
                for link in cols[0].find_all("a"):
                    temp_href = link.get("href")
                    if "txt.gz" in temp_href or "csv" in temp_href or "tar.gz" in temp_href or "tsv" in temp_href or "zip" in temp_href:
                        dataset_href.append(temp_href)
    return dataset_href

def generate_download_url(href):
    download_url = f'https://snap.stanford.edu/data/{href}'
    return download_url

# Function to determine the type of graph based on its description
def classify_graph(description):
    description = description.lower()
    if 'undirected' in description and not 'weighted' in description:
        return 'undirected'
    elif 'directed' in description and not 'weighted' in description:
        return 'directed'
    elif 'weighted' in description and 'directed' in description:
        return 'weighted&directed'
    elif 'weighted' in description and 'undirected' in description:
        return 'weighted&undirected'
    else:
        return 'others'

# Find all dataset entries in the webpage
for table in soup.find_all('table'):
    for row in table.find_all('tr')[1:]:  # Skip the header row
        cols = row.find_all('td')
        if len(cols) == 5:
            dataset_name = cols[0].get_text(strip=True)
            for link in cols[0].find_all("a"):
                href = link.get("href")
            dataset_page = get_page_content(href)
            download_href = get_download_href(dataset_page)
            # print(download_urls)
            if len(download_href) == 0:
                continue
            dataset_type = cols[1].get_text(strip=True)
            dataset_nodes = cols[2].get_text(strip=True)
            dataset_edges = cols[3].get_text(strip=True)
            dataset_description = cols[4].get_text(strip=True) if len(cols) > 4 else ''
            graph_type = classify_graph(dataset_type)
            datasets[graph_type].append({
                'name': dataset_name,
                'type': dataset_type,
                'nodes': dataset_nodes,
                'edges': dataset_edges,
                'download_href': download_href,
                'description': dataset_description
            })
        else:
            continue

dir_paths = []
# Create directories for each graph type
for graph_type, dataset_list in datasets.items():
    for dataset in dataset_list:
        dir_paths.append(os.path.join(graph_type, dataset["name"]))

for path in dir_paths:
    os.makedirs(path, exist_ok=True)

# Function to download a dataset
def download_dataset(dataset_name, download_href, graph_type):
    if "/" in download_href:
        download_href = download_href.split('/')[-1]
        import pdb; pdb.set_trace()
    file_path = os.path.join(graph_type, dataset_name, download_href)
    if os.path.isfile(file_path):
        print(f'File exists at {file_path}, skip downloading it')
    else:
        download_url = generate_download_url(download_href)
        response = requests.get(download_url)
        response.raise_for_status()
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded {download_href} into {graph_type}/{dataset_name}")

def write_dataset_info(dataset_dict, graph_type):
    json_file_path = os.path.join(graph_type, dataset_dict['name'], f"{dataset_dict['name']}.json")
    if not os.path.isfile(json_file_path):
        with open(json_file_path, "w", encoding="utf-8") as file:
            json.dump(dataset_dict, file, indent=4)  # `indent=4` for readable formatting
    else:
        print(f'File exists at {json_file_path}, skip generating graph info json file')

# Download datasets by type
for graph_type, dataset_list in datasets.items():
    for dataset in dataset_list:
        # Construct the download URL based on the dataset name
        write_dataset_info(dataset, graph_type)
        for href in dataset['download_href']:
            try:
                download_dataset(dataset['name'], href, graph_type)
            except requests.HTTPError:
                print(f"Failed to download {dataset['name']} from {href}")


