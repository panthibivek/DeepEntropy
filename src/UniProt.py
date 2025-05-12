import requests
import json


accession = "P01308"
url = f"https://rest.uniprot.org/uniprotkb/{accession}"

headers = {
    "Accept": "application/json"
}


response = requests.get(url, headers=headers)


if response.status_code == 200:
    data = response.json()
    with open(f"{accession}.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"Data saved to {accession}.json")
    try:
        print("Protein name:", data['proteinDescription']['recommendedName']['fullName']['value'])
    except KeyError:
        print("Protein name not found in response.")
else:
    print("Failed to retrieve data:", response.status_code)
