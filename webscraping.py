from bs4 import BeautifulSoup
import requests
import re

countries = [
    "مصر",
    "السعودية",
    "الإمارات",
    "قطر",
    "الكويت",
    "البحرين",
    "سلطنة_عمان",
    "الأردن",
    "سوريا",
    "لبنان"
]

for countryn in countries:
    page = requests.get(f"https://ar.wikipedia.org/wiki/{countryn}")

    def main(page):
        src = page.content
        soup = BeautifulSoup(src, "lxml")
        
        country = soup.find("table", {'class': 'infobox2 infobox'})
        if not country:
            print(f"{countryn}: مش لاقي حاجه")
            return

        rows = country.find_all("tr")
        found_population = False
        found_area = False

        population_number = None
        area_number = None

        for row in rows:
            header = row.find("th")
            data = row.find("td")
            if header and data:
                header_text = header.text.strip()
                data_text = data.text.strip()

                if "المساحة" in header_text or "المساحه" in header_text:
                    matches = re.findall(r'[\d,.]+', data_text)
                    if matches:
                        area_number = float(matches[0].replace(",", "").replace(".", ""))
                        print(f"🔹 مساحة {countryn}: {matches[0]} كم²")
                        found_area = True


                if ("التعداد السكاني" in header_text or "عدد السكان" in header_text) and not found_population:
                    matches = re.findall(r'[\d,]+', data_text)
                    if matches:
                        numbers = [int(m.replace(",", "")) for m in matches]
                        population_number = max(numbers)
                        result = f"{population_number:,}"
                        print(f"🔹 عدد سكان {countryn}: {result} نسمة")
                        found_population = True

        if not found_area:
            print(f"🔹 مساحة {countryn}: مش موجوده")
        if not found_population:
            print(f"🔹 عدد سكان {countryn}: مش موجوده")

        if population_number and area_number and area_number != 0:
            density = round(population_number / area_number, 2)
            print(f"🔹 الكثافة السكانية في {countryn}: {density} نسمة لكل كم²")
        else:
            print(f"🔹 الكثافة السكانية في {countryn}: مش موجوده")

    main(page)
