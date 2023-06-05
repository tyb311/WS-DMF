# Retinal Vessel Segmentation based on Deep Matched Filtering and Supervised Contrastive Learning

By [Yubo Tan](https://github.com/tyb311/)\*


Code is based on ["Self-Supervised Learning with Swin Transformers"](https://arxiv.org/abs/2105.04553). 

## Ideas
- 对比深监督， 不用HRNet或者unet++这一条也用不着了
- focalnce哈哈，真会想，focal应该放在Exp还是log哪？
- dou sigmoid对比衰减系数改为0.1-》0.5-》1
- nmf配合torch匹配加大系数perception

- outlook attention

-	TIP的功能实现依赖于匹配滤波和噪声正则化，可能后者作用更大一些
-	怎么用对比学习完成假阳性排除功能，代替噪声正则化的作用？
-	投影渲染？refine？

-	样本选择策略：
-		假阳性样本不一定只分布在血管边缘，要考虑所有背景像素才行
-		血管周围像素关注的是模糊边界问题
-		背景像素关注的是假阳性样本问题，有效果的话，样本选择划分为三类（可解释性高）



##	思考
-	对比策略究竟有无效果？和dou裸跑比较一下
-	渲染策略有没有效果？tanh重新渲染一下
-	siam停止梯度有没有效果，重新思考一下理论
-	batchsize这些超参数又如何，图片尺寸影响大吗？
-	再换几个模型跑一下试试
-	血管分割不太行，视盘分割怎么样，GAMMA-MICCAI2021比赛


##	反思
-	这两个星期排查了两个错误，以后要经常回顾代码，找出漏洞所在才是啊
1.	样本选择概率排序出错了，后来改正，还学会了index_select函数
2.	不同类别划分，血管、血管边界、血管背景都应该考虑进去
3.	再没有效果，测试下hard意外的其他构造方法
4.	渲染的方法还可以再试试
5.	有没有可能是因为嵌入层（投影位置）太浅了，导致表达网络后面的分割网络学习能力太差


##  sample selection strategy map
### 07-13选择不够细致
def regular(emb, temp=1):
	f, b = torch.chunk(emb, 2, dim=0)
	pos = f @ f.permute(1,0) + b @ b.permute(1,0)
	neg = f @ b.permute(1,0)
	pos = torch.exp(pos/ temp)
	neg = torch.exp(neg/ temp)

	los = -torch.log(pos / (pos + neg + 1e-5)).mean()
	return los
画出来的图明显不具有对称性，这很令人迷惑啊

### 07-13选择不够细致

def regular(emb, temp=1):
	f1,f2, b1,b2 = torch.chunk(emb, 4, dim=0)
	pos = f1 @ f2.permute(1,0) + b1 @ b2.permute(1,0)
	neg = torch.cat([f1,f2],dim=0) @ torch.cat([b1,b2],dim=0).permute(1,0)
	pos = torch.exp(pos/ temp)
	neg = torch.exp(neg/ temp)

	los = -torch.log(pos / (pos + neg + 1e-5)).mean()
	return los



 ###	2021/7/16 23:15:49
甚至是非锐化掩蔽滤波模型加进来，发现没效果，原来是因为做了TTA-flip才提升的分数
一定是哪里又写错了，用上memorybank semihard， 对了，他们使用memorybank肯定是detach的，停止梯度这一点不能放过
latent space之后的分割模块是不是应该加深点，加深之后果然效果好了点
loss_cl.py里面划分倒是没错，但是没有预测器，也没用梯度停止，是不是这个原因才效果不好的啊？

单模形siam两种划分方式2*2比较
如果不划分batch,划分概率效果如何哪？
别管其他的，专攻siam也可以

重新设计损失函数，以及读前列腺论文
把对比分割的代码看一遍，再看transformer论文
如果效果还可以，试试nce3
无奈解决了map不对称的问题，却忘记了加强大的分割器（渲染器），导致效果没有提上去
按照老师要求和论文要求做一些其他工作：
	按照阈值看不同粗细的血管分给效果如何
	看看病变区域分割效果如何


##	SOTA
-	072616drive-siamXsunetXsunetR0.1FCALds5-fr在CAL上超过DRIS，却逊于SUnet


##	2021-07-19
-	发现了siam里面的损失函数部分，数据划分出错了，可能是heatmap不对称的原因
-	明天要跑一下regular3-->V3
-	再跑一下dmf32、align_uniform
-	再解决一下边界模糊+背景假阳性问题
-	深监督+分解对比损失
-	组会时间去 
		loss(pred, proj.detach()) + loss(proj, pred.detach())
		loss(pred, proj.detach()) + loss(pred[:-1:], pred.detach()[:-1:])
-	发现我效果不好的代码，原因可能是：数据增强的部分代码被鸽了！！！！伤心欲绝


##	多数据集
-	可是不同数据库为啥效果还是那么差嘞，解释不通
-	也只是stare数据库上效果不好，哈哈。没有做Leave-One-Out交叉验证
-	解决手段1：加上DiceLoss
-	解决手段2：对比学习或许有效果：采样用hard sampling试试
-	解决手段3：聚类或许有效果
-	解决手段4：困难样本增强



##	训练策略
-	KL散度约束正负样本，效果不行啊，又占用资源
-	使用Re-Drop， 和对比损失一块用效果不行啊
在不同数据集上跑一下，看看结果
为了减少数据预处理运算量，把形态学骨架提取部分写成离线增强的形式
为了弥补现有评价指标的局限性，一边采用FCAL和SS等指标，一边考虑设计新的无参考图像指标


##	模型设计
-	sunet>lunet>punet
siam(无参数attention)


##	损失函数
拿血管骨架的膨胀做损失函数嘞？或者加大困难样本patch的损失权重就可以了吧
vessel aware focal写进去
基于FCAL的损失函数改进：
	dice(conv(pr), dilate(gt))
	dice(conv(pr), dilate(skeleton(gt)))
	要不要skeleton，回调指标不要效果好
下一步：更改血管骨架先验的损失函数，不一定要用单边Dice（我可真会起名字）


##	评价指标 & 回调指标
-	CAL
-	SS
-	阈值下的指标对比
-	用快速fcal四个回调指标过得验证集最佳模型，再考虑别的
-	设计一款比FCAL等任何现有指标都更强的回调指标（即使不能作为损失函数）


##	困难样本增强(iou不现实，CAL？)
困难样本增强，把验证集分数最低的路径加入训练集
针对困难patch单独做匹配滤波引导，或者重点训练



##	采样策略
按照顺序采样得到的map近似对称，也体现出使用hard样本效果更好的思想
按照输出概率范围采样，而不是
想到基于阈值的样本选择，横轴画阈值，纵轴画随机采样的浓度
-	极端样本选择策略
-	形态学开运算可以找到细血管位置，然后针对细血管定制对比学习采样策略
doverconv
为了保证分割细血管，不顾假阳性，只干掉假阴性（后期考虑非线条形状假阳性样本）



##	对比学习
-	python画出投影超球面，投影到三维也没啥意思，还是T-SNE投影到二维吧
对比学习在困难patch上面进行会不会更好
根据map画出颜色深浅代表iou的箭头图，确定颜色最深的路线
观察对比学习是否真的优化了边缘，去除了假阳性
停止梯度，sim  mse齐上阵
nce2 vs nce3, sim2 vs sim3, sim3是否用sim2实现
-	突然发现：sample没有考虑batchsize，无论batchsize多大，采样点都是恒定数量的
	这样的话，简单点的方法是增加采样点数，麻烦点的方法是重写采样函数

用hard试试sim nice nce2再考虑别的
样本取太少了
血管和背景像素选择策略不该一样
-	重点是啥？是把细血管和周围环境分开
-	再有是不是超平面维度不够高？以至于影响了特征
-	有没有可能在超球面，预测器之前做L2 Loss
-	或者对feat做batchnorm，稍微把像素空间和向量空间对齐
 2021/7/29 23:05:34
兴许是维度太高了，导致计算速度慢
把骨架先验模块向前移动，约束倒数第二层特征
明天一定好好调整dmf和对比投影




##	改进DRIS
深度匹配滤波还是要加把油啊！！！
单纯把匹配滤波作为一种attention如何哪？匹配滤波未必就一定是完美的呀！是吧

还是dris，但是全靠匹配滤波貌似也不现实，至少现在看来是这样的
match filter prior or guided
关键是把细血管检测出来，形状先验，匹配引导或者基于分解机制的细节增强三种方案
匹配核通过随机投影约束，就避免了手工设计魔板的问题，vgg16就相当于一种随机投影
考虑这样的思路：
	对卷积做L1+L2约束，整理成轴向高斯模样
	对响应60层做同样的约束
	对最大响应做TV+Perception，这里尽管用损失函数
	投影到超球面，做对比学习
	超球面阈值射影，获得分割结果



##	暑假研究方向
-	搞完眼底图像
	MCU:	match filter guided contrastive learning two unet
	MCUV2:	deep match filter & contrastive sphere peojection
	基于骨架先验的。。。
	基于监督对比学习的。。。
	基于深度匹配滤波的无监督。。。（结合纹理损失）
	无参考的血管分割指标（粗血管用像素级，细血管用纹理级别）

-	重新设计个对比损失函数去做超声图像分割
-	Mobile Transformer HAR
-	图卷积+水平集

[宁波工研院]https://imed.nimte.ac.cn/data-index.html