from __future__ import print_function
import logging

import grpc

import memegenerator_pb2
import memegenerator_pb2_grpc
import sqlite3
from sqlite3 import Error
import time
from classes.predictor import Predictor

class Client:
    conn = None

    def get_url(self, caption, templateid):
        # with grpc.insecure_channel('localhost:50051') as channel:
        with grpc.insecure_channel('api:50051') as channel:
            stub = memegenerator_pb2_grpc.MemerStub(channel)
            response = stub.GetMemeUrl(

                memegenerator_pb2.MemeRequest(
                    caption=caption, memeid=templateid))
        print("Memer client received url: " + response.url)
        return response.url

    def create_connection(self, db_file):
        """ create a database connection to the SQLite database
            specified by the db_file
        :param db_file: database file
        :return: Connection object or None
        """
        try:
            self.conn = sqlite3.connect(db_file)
            print(sqlite3.version)
        except Error as e:
            print(e)

    def select_memes_for_imgflip(self):
        """
        Query all rows in the tasks table
        :param conn: the Connection object
        :return:
        """
        # cur = self.conn.cursor()
        # cur.execute("SELECT * FROM sqlite_master")
        # rows = cur.fetchall()
        # for row in rows:
        #     print(row)
        cur = self.conn.cursor()
        cur.execute("SELECT id,templateid,caption FROM memes_meme WHERE url is null and templateid is not null and caption is not null")
        rows = cur.fetchall()
        return rows

    def update_meme_url(self, id_url):
        """
        :param id_url:
        :return: id
        """
        sql = ''' UPDATE memes_meme
                SET url = ?
                WHERE id = ?'''
        cur = self.conn.cursor()
        cur.execute(sql, id_url)
        self.conn.commit()

    def update_memes_url(self, rows):
        print(rows)
        ids_urls = [(self.get_url(row[2], row[1]), row[0])
                    for row in rows]
        # get_url('when implementing grpc|is hard', "102156234")
        print(ids_urls)
        for id_url in ids_urls:
            self.update_meme_url(id_url)

    def select_memes_for_caption(self):
        """
        Query all rows in the tasks table
        :param conn: the Connection object
        :return:
        """
        cur = self.conn.cursor()
        cur.execute("SELECT id,templateid,caption FROM memes_meme WHERE url is null and templateid is not null and caption is null")
        rows = cur.fetchall()
        return rows

    def update_meme_caption(self, caption, url, id):
        """
        :param caption, url, id:
        :return: id
        """
        sql = ''' UPDATE memes_meme
                SET caption = ?,
                url = ?
                WHERE id = ?'''
        cur = self.conn.cursor()
        print(f'caption:{caption}, url:{url}, id:{id}')
        cur.execute(sql, (caption, url, id))
        self.conn.commit()

    def update_memes_caption(self, rows, predictor):
        for row in rows: # id, templateid, caption 
            print(f'Generate caption for meme {row[0]}')
            caption = predictor.predict(row[1])
            print(f'Prediction {caption}')
            url = self.get_url(caption, row[1])
            id = row[0]
            self.update_meme_caption(caption, url, id)

    def run(self, predictor):
        try:
            print('Enabling db connection')
            logging.basicConfig()
            self.create_connection("./db.sqlite3")
            while True:
                print('Enter loop')
                print(self.conn)

                rows = self.select_memes_for_imgflip()
                self.update_memes_url(rows)

                if predictor is not None:
                    rows = self.select_memes_for_caption()
                    self.update_memes_caption(rows, predictor)

                time.sleep(10)
        except Error as e:
            print('Error')
            print(e)
        finally:
            try:  # double try block for Keyboard intrerrupt
                if self.conn:
                    self.conn.close()
                    print('Session closed')
            except Error as e:
                self.conn.close()


if __name__ == '__main__':
    print('Loading predictor')
    predictor = Predictor()
    # predictor.generateCaptions()
    print('Predictor loaded')
    client = Client()
    print('Predictor created')
    client.run(predictor)
