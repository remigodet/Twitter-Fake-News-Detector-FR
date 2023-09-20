# adding labels to our data using real neurons !
import tkinter as tk
import os
# import sys
# sys.path.append(".")


class App(tk.Frame):
    def __init__(self, root):
        tk.Frame.__init__(self, root)
        self.parent = root
        self.idx = None
        self.data = None
        self.nb_twits = 0
        self.create_first_frame()
        self.create_frame()
        self.first_frame.tkraise()

    def create_first_frame(self):
        self.first_frame = tk.Frame(self)
        self.info = tk.Label(self.first_frame, text='Ready to label ?')
        self.info.pack(side=tk.TOP)
        self.go = tk.Button(self.first_frame,
                            text='Go !',
                            command=self.next_twit)
        self.go.pack()
        self.first_frame.grid(row=0, column=0, sticky="nsew")
        self.pack()

    def create_frame(self):
        self.twit_frame = tk.Frame(self)
        self.twit = 'no tweet yet :/'
        # nb tweet
        self.info2 = tk.Label(
            self.twit_frame, text='please scrape for tweets:')
        self.info2.pack(side=tk.TOP)
        # tweet
        self.info = tk.Label(
            self.twit_frame, text=self.twit, wraplength=350)
        self.info.pack(side=tk.TOP)

        self.buttonwidget = tk.Frame(self.twit_frame)

        self.pos = tk.Button(self.buttonwidget,
                             text='TRUE',
                             command=lambda: label('true'))
        self.pos.grid(row=0, column=0)

        self.neg = tk.Button(self.buttonwidget,
                             text='FAKE',
                             command=lambda: label('fake'))
        self.neg.grid(row=0, column=2)

        self.neu = tk.Button(self.buttonwidget,
                             text='NEU',
                             command=lambda: label('neu'))
        self.neu.grid(row=0, column=1)

        self.reset = tk.Button(self.buttonwidget,
                               text='Reset',
                               command=lambda: label('oops'))
        self.reset.grid(row=2, column=1)

        self.passover = tk.Button(self.buttonwidget,
                                  text='Passover',
                                  command=lambda: label('passover'))
        self.passover.grid(row=1, column=1)
        # arrow keys binding
        self.buttonwidget.bind('<Left>', lambda self: label('true'))
        self.buttonwidget.bind('<Up>', lambda self: label('neu'))
        self.buttonwidget.bind('<Right>', lambda self: label('fake'))
        self.buttonwidget.bind('<Down>', lambda self: label('passover'))
        self.buttonwidget.focus_set()

        self.buttonwidget.pack(side=tk.BOTTOM)

        self.twit_frame.grid(row=0, column=0, sticky="nsew")

        def save_label(label):
            with open(f"NLP/labeled_data/{label}/{self.file}{self.idx}.txt", "w", encoding="utf-8") as file:
                print(self.data['tweet_textual_content'][self.idx])
                file.write(
                    self.data['tweet_textual_content'][self.idx]
                )

        def label(label):
            if label == 'oops':
                self.idx -= 1
                os.remove(
                    f"NLP/labeled_data/{label}/{self.file}{self.idx}.txt")
                self.idx -= 1

            elif label == 'passover':
                print('tweet passed :/')
            else:
                save_label(label)
            self.idx += 1

            self.next_twit()

    def next_twit(self):

        if self.idx == None:
            self.idx = 0
        elif self.idx < 0:
            self.idx = 0
        elif self.idx == self.nb_twits-1:
            App.destroy
        print(self.idx)
        self.twit_frame.tkraise()
        self.info.config(text=self.data['tweet_textual_content'][self.idx])
        self.info2.config(text='tweet {} out of {}'.format(
            self.idx+1, self.nb_twits))

    def get_data(self, file, data):
        self.file = file
        self.data = data
        self.nb_twits = len(data)
