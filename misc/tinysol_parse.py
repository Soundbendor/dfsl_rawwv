import os,sys,csv
sys.path.insert(0, os.path.dirname(os.path.split(__file__)[0]))
import util.globals as UG

inst_family = ["Brass", "Strings", "Winds"]
ret = []

brass_dir = os.path.join(UG.DEF_TINYSOLDIR, "Brass")
str_dir = os.path.join(UG.DEF_TINYSOLDIR, "Strings")
winds_dir = os.path.join(UG.DEF_TINYSOLDIR, "Winds")
out_f = os.path.join(UG.DEF_TINYSOLDIR, "tinysol.csv")
cur_prefix = os.path.join(UG.DEF_TINYSOLDIR, os.sep)

get_rel_dir = lambda x: x.removeprefix(UG.DEF_TINYSOLDIR).removeprefix("/")
for cur_fam in inst_family:
    fam_dir = os.path.join(UG.DEF_TINYSOLDIR, cur_fam)
    inst_list = os.listdir(fam_dir)
    for cur_inst in inst_list:
        inst_dir = os.path.join(fam_dir, cur_inst)
        technique_list = os.listdir(inst_dir)
        for cur_technique in technique_list:
            note_dir = os.path.join(inst_dir, cur_technique)
            snd_list = os.listdir(note_dir)
            #print(note_list)
            for cur_snd in snd_list:
                snd_split = cur_snd.split("-")
                snd_inst = snd_split[0]
                snd_tech = snd_split[1]
                snd_pitch = snd_split[2]
                snd_split2 = snd_split[3].split(".")
                snd_dyn = snd_split2[0]
                snd_fullpath = os.path.join(note_dir, cur_snd)
                snd_path = get_rel_dir(snd_fullpath)
                snd_info = {"instrument": snd_inst, "technique": snd_tech, "pitch": snd_pitch, "dyn": snd_dyn, "path": snd_path}
                ret.append(snd_info)
#print(ret)

with open(out_f, "w", newline="") as csvf:
    fieldnames = ["instrument", "technique", "pitch", "dyn", "path"]
    csvw = csv.DictWriter(csvf, fieldnames=fieldnames)
    csvw.writeheader()
    for entry in ret:
        csvw.writerow(entry)
