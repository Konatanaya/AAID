class Message:
    def __init__(self, topic, sender, time_step):
        self.topic = topic
        self.sender = sender
        self.time_step = time_step

    def __eq__(self, other):
        if type(other) == type(self) and \
                other.topic == self.topic and \
                other.sender == self.sender:
            return True
        else:
            return False

    def __hash__(self):
        data = str(self.topic) + str(self.sender)
        return hash(data)


if __name__ == '__main__':
    msg = Message(1, 1024, 0)
    msg2 = Message(1, 1024, 0)
    list = [msg, msg2]
    print(set(list)[0].sender)
