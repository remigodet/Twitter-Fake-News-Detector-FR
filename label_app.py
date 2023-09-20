from data import get_data, scrape
from NLP.brainpower import App
import tkinter as tk


def label_tweets(file):
    data = get_data(file+".json")
    root = tk.Tk()
    root.geometry('400x200')
    myapp = App(root)
    myapp.get_data(file, data)
    root.mainloop()


def start_label(keywords, n):
    # comment this if you don't want to scrape
    scrape([keywords], n, user)
    # do not comment
    label_tweets(keywords)


query = input("enter keywords or # or @: \n")
n = int(input("n="))
user = bool(input("user : True/False\n"))

start_label(query+"", n)
