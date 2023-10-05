import pprint
from urllib.parse import urljoin, urlparse  # Import urljoin function

import pandas as pd
import requests
from bs4 import BeautifulSoup


# Try this, please: "https://www.yelp.ca/search?find_desc=Restaurants&find_loc=Mississauga%2C+Ontario"

def get_all_page_urls(search_url):
    # Send an HTTP GET request
    response = requests.get(search_url)

    # Check if the request was successful
    if response.status_code != 200:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")

    # Parse the HTML content of the page
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all elements with the specified class name
    business_names = soup.find_all(class_="businessName__09f24__EYSZE css-1jq1ouh")

    # Extract and print the fully qualified URLs, for now i have defined
    base_url = get_base_url(search_url)  # "https://www.yelp.ca"  # Base URL
    hrefs = []
    for business_name in business_names:
        a_tag = business_name.find('a')
        if a_tag and 'href' in a_tag.attrs:
            fully_qualified_url = urljoin(base_url, a_tag['href'])  # Combine base_url and href
            hrefs.append(fully_qualified_url)

    return hrefs


def get_base_url(search_url):
    parsed_url = urlparse(search_url)

    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    print("Base url - ", base_url)
    # Get the base URL
    return base_url


def get_review_dict(rest_url):
    response = requests.get(rest_url)

    # Check if the request was successful
    if response.status_code != 200:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")

    # Parse the HTML content of the page
    soup = BeautifulSoup(response.text, 'html.parser')

    restaurant_name_element = soup.find(class_="photoHeader__09f24__nPvHp css-1qn0b6x")
    restaurant_name = restaurant_name_element.find('h1').text.strip()

    # Extract total reviews
    total_reviews_element = soup.find(class_="rating-text__09f24__VDRkR css-ea1vb8")
    total_reviews = total_reviews_element.find('p').text.strip()

    # Find the reviews section
    reviews = soup.find(id="reviews")

    # Find all review items
    review_items = reviews.find_all(class_="css-1q2nwpv")

    # Iterate through review items

    reviews = []
    for i in range(1, len(review_items)):  # Start from 1 to skip the first element
        review_box = review_items[i].find(class_="css-1qn0b6x")

        # Extract user information
        user_info = review_box.find(class_="user-passport-info")
        name = user_info.find('a').text

        # Extract review text
        review = review_box.find('p').text

        # Extract rating
        rating = review_box.find(class_="five-stars__09f24__mBKym five-stars--regular__09f24__DgBNj css-1jq1ouh")[
            "aria-label"]

        # Print the extracted information

        review_data = {
            "Restaurant Url": rest_url,
            "Restaurant Name": restaurant_name,
            "Total reviews": total_reviews,
            "Name": name,
            "Review": review,
            "Rating": rating
        }

        reviews.append(review_data)
    return reviews


if __name__ == '__main__':
    search_url = input(
        "Enter search url (you can try this - https://www.yelp.ca/search?find_desc=Restaurants&find_loc=Mississauga%2C+Ontario): ")
    hrefs = get_all_page_urls(search_url.strip())

    # Not paged, we can do paging here as well if we have time.
    print("Business listed on this search page - ")
    pprint.pprint(hrefs)

    all_reviews = []
    for href in hrefs:
        print("Scraping - ", href)
        try:
            reviews = get_review_dict(href)
            all_reviews.extend(reviews)
            pprint.pprint(reviews)
        except:
            print("Error while parsing - ", href)

    pd.DataFrame(all_reviews).to_csv("reviews.csv")
