import pymysql.cursors


conn = pymysql.connect(host='127.0.0.1',
                       user='shimada',
                       password='nshimada!123',
                       db='tabelog_production',
                       charset='utf8mb4',
                       cursorclass=pymysql.cursors.DictCursor)

try:
    with conn.cursor() as cursor:
        # sql = "SELECT name, kids_notice FROM restaurants WHERE address1 like '%新宿区%' AND kids_notice!=''"  # like '%東京都%'

        # sql = "SELECT restaurants.name, COUNT(DISTINCT restaurant_menus.name) FROM restaurants"
        # sql += " JOIN restaurant_menus ON restaurants.id=restaurant_menus.restaurant_id"
        # sql += " WHERE restaurants.address1 like '%新宿区%'"
        # sql += " GROUP BY restaurants.id"

        # sql = "SELECT restaurants.name, COUNT(DISTINCT review_coms.comment) FROM restaurants"
        # sql += " JOIN"
        # sql += " (SELECT reviews.restaurant_id AS restaurant_id, review_comments.comment AS comment"
        # sql += " FROM reviews JOIN review_comments ON reviews.id=review_comments.review_id) AS review_coms"
        # sql += " ON restaurants.id=review_coms.restaurant_id"
        # sql += " WHERE restaurants.address1 like '%新宿区%'"
        # sql += " GROUP BY restaurants.id"

        sql = "SELECT name, address1 FROM restaurants WHERE name LIKE '%鱧%料%理%三%栄%'"

        cursor.execute(sql)
        result = cursor.fetchall()
        print(result)
finally:
    conn.close()
