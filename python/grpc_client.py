from __future__ import print_function
import logging

import grpc

import memegenerator_pb2
import memegenerator_pb2_grpc
import sqlite3
from sqlite3 import Error
import time


class Client:
    conn = None

    def get_url(self, caption, id):
        with grpc.insecure_channel('localhost:50051') as channel:
            stub = memegenerator_pb2_grpc.MemerStub(channel)
            response = stub.GetMemeUrl(

                memegenerator_pb2.MemeRequest(
                    caption=caption, memeid=id))
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

    def select_all_memes(self):
        """
        Query all rows in the tasks table
        :param conn: the Connection object
        :return:
        """
        cur = self.conn.cursor()
        cur.execute("SELECT id,templateid,caption FROM memes_meme WHERE url=''")
        rows = cur.fetchall()
        return rows

    def update_meme(self, id_url):
        """
        :param id_url:
        :return: project id
        """
        sql = ''' UPDATE memes_meme
                SET url = ?
                WHERE id = ?'''
        cur = self.conn.cursor()
        cur.execute(sql, id_url)
        self.conn.commit()

    def run(self):
        try:
            logging.basicConfig()
            self.create_connection("../backend/db.sqlite3")
            while True:
                rows = self.select_all_memes()
                print(self.conn)
                for row in rows:
                    print(
                        f'Updating row id {row[0]} templateid {row[1]} caption {row[2]}')
                ids_urls = [(self.get_url(row[2], row[1]), row[0])
                            for row in rows]
                for id_url in ids_urls:
                    self.update_meme(id_url)
                # get_url('when implementing grpc|is hard', "102156234")
                time.sleep(10)
        except Error as e:
            print(e)
        finally:
            try:  # double try block for Keyboard intrerrupt
                if self.conn:
                    self.conn.close()
                    print('Session closed')
            except Error as e:
                self.conn.close()


if __name__ == '__main__':
    client = Client()
    client.run()
