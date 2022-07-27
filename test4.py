import os

def switching_od_label(src_path, dst_path):
    # src_path txt 파일 읽어서 dst_path에 저장
    # 0 <-> 4 , 1 <-> 3 라벨값 바꾸는 함수

    for fname in os.listdir(src_path):
        print(fname)
        f = open(os.path.join(src_path,fname), 'r')
        lines = f.readlines()
        switching_lines = ""
        for line in lines:
            print(line, end='')
            # switching
            switching_lines += str(4 - int(line[0]))
            switching_lines += line[1:]
        f.close()

        print('switching_lines')
        print(switching_lines)
        f = open(os.path.join(dst_path, fname), 'w')
        f.write(switching_lines)
        f.close()
    pass

if __name__ == '__main__':
    switching_od_label('C:\\Users\\정희운\\Desktop\\label', 'C:\\Users\\정희운\\Desktop\\temp')