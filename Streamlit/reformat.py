import json

def reformat_location_data(data):
    reformatted_locations = []
    for location in data["Locations"]:
        reformatted_location = {
            "LocationID": location["LocationID"],
            "LocationName": location["LocationName"],
            "DisplayName": location["DisplayName"],
            "Coordinates": {
                "Latitude": location["Latitude"],
                "Longitude": location["Longitude"]
            },
            "Address": {
                "AddressLine1": location["AddressLine1"],
                "AddressLine2": location["AddressLine2"],
                "City": location["City"],
                "StateProvince": location["StateProvince"],
                "PostalCode": location["PostalCode"],
                "Country": location["Country"]
            },
            "Contact": {
                "Telephone": location["Telephone"],
                "Fax": location.get("Fax", ""),
                "Email": location.get("Email", "")
            },
            "OperatingHours": {
                "WeeklyOperatingDays": location["WeeklyOperatingDays"],
                "WeeklyOperatingHours": location["WeeklyOperatingHours"],
                "HoursOfOperation24": location["HoursOfOperation24"]
            },
            "AdditionalInfo": {
                "TimeZone": location["TimeZone"],
                "Products": [product.strip(";") for product in location["Products"].strip("[]").split("]|[")],
                "Franchise": location["Franchise"],
                "Category": location["Category"],
                "Promotions": location["Promotions"],
                "BuyOnline": location["BuyOnline"],
                "WebsiteUrl": location["WebsiteUrl"]
            }
        }
        reformatted_locations.append(reformatted_location)
    
    return {"Locations": reformatted_locations}

def save_reformatted_data_to_txt(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def main():
    input_filename = 'Data/Data/location.txt'
    output_filename = 'reformatted_location.txt'

    with open(input_filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    reformatted_data = reformat_location_data(data)
    save_reformatted_data_to_txt(reformatted_data, output_filename)

if __name__ == "__main__":
    main()