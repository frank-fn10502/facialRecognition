import configparser as cfg


myCfg = cfg.ConfigParser()
myCfg.add_section('Default')
myCfg.set('Default' ,'device' ,'0')        #裝置
myCfg.set('Default' ,'deviceW' ,'1920')    #裝置width (嘗試設置) 超過最高解析度會自動設定為攝影機的最高解析度
myCfg.set('Default' ,'deviceH' ,'1080')    #裝置height(嘗試設置)

myCfg.set('Default' ,'maxFaceNum' ,'10')   #要存幾張人臉
myCfg.set('Default' ,'saveInterval','0.5') #儲存間隔 幾秒

myCfg.set('Default' ,'jsonFilePath' ,'./json/detectionData.json')#未完成 文件需要創造
myCfg.set('Default' ,'writeJson' ,'True')#是否要寫檔



myCfg.add_section('DectionCore')
myCfg.set('DectionCore' ,'thresh' ,'0.6')

myCfg.add_section('RecognitionCore')
myCfg.set('RecognitionCore' ,'distTresh' ,'0.85')# < 1
myCfg.set('RecognitionCore' ,'imageFolder' ,'./Image')  #圖片資料的儲存位置

with open('./cfg/myConfig.cfg' ,'w' ,encoding='utf-8') as cf:
    myCfg.write(cf)