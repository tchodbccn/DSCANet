from runx.logx import logx
from config import cfg
from Utils import getCurrentTime, makedirs

mylog = logx

cfg.STARTTIME = getCurrentTime()


logdir = './logs/' + cfg.STARTTIME
makedirs(logdir)


mylog.initialize(logdir=logdir, tensorboard=True)
mylog.msg("start time：" + cfg.STARTTIME)