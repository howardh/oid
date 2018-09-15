from io import BytesIO

import data
from data.classdescriptions import ClassDescriptions

class_description_csv_40 = """/m/0100nhbf,Sprenger's tulip
/m/0104x9kv,Vinegret
/m/0105jzwx,Dabu-dabu
/m/0105ld7g,Pistachio ice cream
/m/0105lxy5,Woku
/m/0105n86x,Pastila
/m/0105ts35,Burasa
/m/0108_09c,Summer snowflake
/m/01_097,Airmail
/m/010dmf,Isle of man tt
/m/010jjr,Amusement park
/m/010l12,Roller coaster
/m/010lq47b,Witch hat
/m/010ls_cv,Sandwich Cookies
/m/01_0wf,Common Nighthawk
/m/010xc,Aspartame
/m/01127,Air conditioning
/m/01_12b,Granny smith
/m/0114n,Atari jaguar
/m/01154,Atari lynx
/m/011_6p,Kazoo
/m/0117_25k,Saffron crocus
/m/01172_8x,Pencil skirt
/m/0117wzjg,Zenvo ST
/m/0117z,Air show
/m/0118b5n4,May day
/m/0118ms9c,Reflex camera
/m/0118n_9r,Water bottle
/m/0118n_nl,Unleavened bread
/m/0118q29r,Ides of march
/m/01195jk4,Jean short
/m/0119x1zy,Bun
/m/0119x27p,Cocker spaniel
/m/011b3pkg,Giant freshwater stingray
/m/011b986k,Rockhopper penguin
/m/011bc8hg,Camomile
/m/011bfkzx,Beaglier
/m/011_dp,Membranophone
/m/011_f4,String instrument
/m/011_g9,Wind instrument"""

def test_load(tmpdir):
    input_dir = tmpdir.mkdir('input')
    output_dir = tmpdir.mkdir('output')
    f = input_dir.join('class-descriptions.csv')
    f.write(class_description_csv_40)
    cd = ClassDescriptions(input_dir=input_dir.strpath, output_dir=output_dir.strpath)
    cd.load()
    # First row
    assert cd['/m/0100nhbf'] == 'Sprenger\'s tulip'
    assert cd['Sprenger\'s tulip'] == '/m/0100nhbf'
    # Last row
    assert cd['/m/011_g9'] == 'Wind instrument'
    assert cd['Wind instrument'] == '/m/011_g9'
    # Num rows
    assert len(cd) == 41, 'Incorrect number of rows loaded'
