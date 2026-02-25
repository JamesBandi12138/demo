import requests
from bs4 import BeautifulSoup
import csv
import time

ua = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
url = 'https://movie.douban.com/top250'

movies = []

for page in range(10):
    resp = requests.get(url, params={'start': page*25}, headers={'User-Agent': ua})
    soup = BeautifulSoup(resp.text, 'html.parser')
    items = soup.select('.grid_view li')
    
    for item in items:
        movie = {}
        movie['rank'] = item.select_one('em').text
        movie['title'] = item.select_one('.title').text
        movie['rating'] = item.select_one('.rating_num').text
        movie['info'] = ' '.join(item.select_one('.bd p').text.split())
        
        quote = item.select_one('.inq')
        movie['quote'] = quote.text if quote else ''
        
        movie['link'] = item.select_one('a')['href']
        movies.append(movie)
    
    print(f'第{page+1}页 done')
    time.sleep(1)

with open('douban_top250.csv', 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=['rank','title','rating','info','quote','link'])
    writer.writeheader()
    writer.writerows(movies)

print(f'搞定，共{len(movies)}条')
