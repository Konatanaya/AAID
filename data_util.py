def remove_seperator(file, seperator):
    with open('./dataset/' + file + '.txt') as f, open('./dataset/' + file + '_clean.txt', 'w') as new:
        for line in f:
            arr = line.split(seperator)
            newline = arr[0] + ' ' + arr[1] + '\n'
            new.write(newline)


def add_new_id(file, seperator):
    with open('./dataset/' + file + '.txt') as f, open('./dataset/' + file + '_clean.txt', 'w') as new:
        i = 0
        for line in f:
            arr = line.split(seperator)
            newline = str(i) + ' ' + arr[0] + '\n'
            new.write(newline)
            i = i + 1

def change_to_new_id(destination, reference):
    with open('./dataset/' + destination + '.txt') as d, open('./dataset/' + reference + '.txt') as r, open('./dataset/' + destination + '_clean.txt', 'w') as new:
        dic = {}
        arr = []
        for l in r:
            n = l.split(' ')[0]
            o = l.split(' ')[1].strip()
            dic[o] = n
        print(dic)
        for line in d:
            # column 1
            old_id_u = line.split(' ')[0]
            new_id_u = dic.get(old_id_u)
            old_id_v = line.split(' ')[1].strip()
            new_id_v = dic.get(old_id_v)
            if new_id_v is None:
                arr.append(old_id_v)
            else:
                newline = new_id_u + ' ' + new_id_v + '\n'
                new.write(newline)
        print(arr)
        # append_receiver_index(arr)


def append_receiver_index(arr):
    with open('./dataset/userstatistic_clean_clean.txt', 'w') as new:
        index = 10994
        for i in arr:
            new_line = str(index) + ' ' + i + '\n'
            new.write(new_line)
            index = index + 1


if __name__ == '__main__':
    # remove_seperator('ciao', '::::')
    # add_new_id('userstatistic', '::::')
    change_to_new_id('ciao_clean', 'userstatistic_clean')