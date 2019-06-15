import os


def convert_avs_to_avi(avs_file, avi_file):
    file = open("temp.vdub", "w")
    file.write("VirtualDub.Open(\"%s\");" % avs_file)
    file.write("VirtualDub.video.SetCompression(\"cvid\", 0, 10000, 0);")
    file.write("VirtualDub.SaveAVI(\"%s\");" % avi_file)
    file.write("VirtualDub.Close();")
    file.close()

    os.system("C:\\ProgramData\\chocolatey\\lib\\virtualdub\\tools\\vdub64.exe /i temp.vdub")

    os.remove("temp.vdub")