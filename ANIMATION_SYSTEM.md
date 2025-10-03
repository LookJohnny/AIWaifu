# Ani 3D动画系统完整指南

## ✅ 已实现的功能

### 1. 自然站立姿势
- ✅ 修复 T-pose（双手平举）
- ✅ 手臂自然垂放在身体两侧
- ✅ 轻微弯曲的肘部
- ✅ 自然的手部角度
- ✅ 微微倾斜的臀部姿态

### 2. 多层次空闲动画系统

#### 呼吸效果
- 胸部扩张收缩（1.2Hz频率）
- 上胸部和下胸部分层动画
- 微妙但明显的生命感

#### 重心转移
- 臀部左右微微摇摆（0.4Hz）
- 脊柱反向补偿
- 模拟真实的重量分配

#### 头部自然运动
- 左右查看（0.35Hz）
- 微微倾斜（0.28Hz）
- 上下点头（0.42Hz）
- 颈部跟随运动

#### 手臂微动作
- 手臂轻微摆动（0.6Hz）
- 基于基础姿势的变化
- 保持自然垂放的同时添加生命感

#### 手指细节
- 手指微妙摆动（2.0Hz）
- 左右手反向运动
- 增加真实性

### 3. 丰富的手势系统

#### 可用手势
1. **wave** - 挥手打招呼
   - 右手抬起挥动
   - 4次循环波动
   - 持续1.5秒

2. **nod** - 点头
   - 头部和颈部协同
   - 2次上下点头
   - 表示同意/理解

3. **tilt** - 歪头
   - 头部侧向倾斜
   - 表示疑惑/可爱

4. **smallWave** - 小幅挥手
   - 前臂小幅度摆动
   - 3次快速波动
   - 更含蓄的招呼

5. **think** - 思考姿势
   - 头部侧向
   - 右手抬起接近脸部
   - 表示思考中

6. **shrug** - 耸肩
   - 双肩上抬
   - 手臂同时抬起
   - 表示不确定/无奈

### 4. 说话时的动态表现

#### 自动手势触发
- 说话时每2秒随机触发一个手势
- 从 nod, tilt, smallWave 中随机选择
- 让对话更生动自然

#### 嘴部动画
- 5个基础口型循环（A, I, U, E, O）
- 100ms间隔切换
- 与音频播放同步

### 5. 表情系统
- joy（开心）→ happy 表情
- sad（难过）→ sad 表情
- anger（生气）→ angry 表情
- surprise（惊讶）→ surprised 表情
- neutral（中性）→ neutral 表情

## 🎯 测试和调整

### 主界面测试
打开: **http://localhost:8000**

**测试空闲动画：**
1. 等待角色加载
2. 观察呼吸、头部摇摆、手臂微动
3. 角色应该看起来"活着"而不是雕像

**测试对话动画：**
1. 点击 "Hold to Talk" 说话
2. 观察说话时的手势
3. 观察嘴部同步
4. 观察表情变化

### 姿势调整工具
打开: **http://localhost:8000/pose**

**功能：**
- 实时调整手臂旋转角度
- 左右臂独立控制
- 上臂 X, Y, Z 三轴旋转
- 下臂 Z 轴旋转（弯曲）

**使用步骤：**
1. 用滑块调整角度
2. 实时查看效果
3. 点击 "Copy Values to Console"
4. 在控制台（F12）获取代码
5. 复制到 avatar_3d.html 中

### 当前默认值
```javascript
// 左臂
leftUpperArm.rotation.set(0.3, 0, 0.8);
leftLowerArm.rotation.set(0, 0, 0.3);

// 右臂
rightUpperArm.rotation.set(0.3, 0, -0.8);
rightLowerArm.rotation.set(0, 0, -0.3);
```

## 🔧 调整建议

### 如果手臂还是太高
增大 Z 值（向下旋转更多）：
```javascript
leftUpperArm.rotation.set(0.3, 0, 1.2);   // 从 0.8 增加到 1.2
rightUpperArm.rotation.set(0.3, 0, -1.2); // 从 -0.8 增加到 -1.2
```

### 如果手臂太低
减小 Z 值：
```javascript
leftUpperArm.rotation.set(0.3, 0, 0.5);   // 从 0.8 减少到 0.5
rightUpperArm.rotation.set(0.3, 0, -0.5); // 从 -0.8 减少到 -0.5
```

### 如果想手臂更向前
调整 X 值：
```javascript
leftUpperArm.rotation.set(0.5, 0, 0.8);   // X 从 0.3 增加到 0.5
rightUpperArm.rotation.set(0.5, 0, -0.8);
```

### 如果手肘太僵硬
增加下臂弯曲：
```javascript
leftLowerArm.rotation.set(0, 0, 0.5);   // 从 0.3 增加到 0.5
rightLowerArm.rotation.set(0, 0, -0.5);
```

## 📊 动画强度调整

### 减少动画幅度（更含蓄）
在 `avatar_3d.html` 中修改：

```javascript
// 呼吸强度（行 639）
const breathCycle = Math.sin(bodyAnimationTime * 1.2) * 0.01;  // 从 0.015 减少到 0.01

// 重心转移（行 650）
const swayCycle = Math.sin(bodyAnimationTime * 0.4) * 0.02;  // 从 0.03 减少到 0.02

// 头部运动（行 662-664）
const headSway = Math.sin(bodyAnimationTime * 0.35) * 0.05;  // 从 0.08 减少到 0.05
const headTilt = Math.sin(bodyAnimationTime * 0.28) * 0.02;  // 从 0.03 减少到 0.02
const headNod = Math.sin(bodyAnimationTime * 0.42) * 0.03;   // 从 0.04 减少到 0.03
```

### 增加动画幅度（更夸张）
```javascript
// 呼吸强度
const breathCycle = Math.sin(bodyAnimationTime * 1.2) * 0.02;  // 增加到 0.02

// 重心转移
const swayCycle = Math.sin(bodyAnimationTime * 0.4) * 0.05;  // 增加到 0.05

// 头部运动
const headSway = Math.sin(bodyAnimationTime * 0.35) * 0.12;  // 增加到 0.12
```

## 🎨 手势触发

### 手动触发手势（浏览器 Console）
```javascript
// 挥手
ani.playGesture('wave')

// 点头
ani.playGesture('nod')

// 歪头
ani.playGesture('tilt')

// 小挥手
ani.playGesture('smallWave')

// 思考
ani.playGesture('think')

// 耸肩
ani.playGesture('shrug')
```

### 修改说话时的手势
在 `avatar_3d.html` 行 516 修改：
```javascript
const gestures = ['nod', 'tilt', 'smallWave'];  // 添加或删除手势
```

### 调整手势频率
在 `avatar_3d.html` 行 519 修改：
```javascript
}, 2000);  // 从 2000ms 改为 3000ms（更少手势）或 1000ms（更多手势）
```

## 🎭 表情与情绪联动

### 当前映射
```javascript
const EMOTION_MAP = {
    'joy': 'happy',      // 开心 → 微笑
    'sad': 'sad',        // 难过 → 皱眉
    'anger': 'angry',    // 生气 → 愤怒
    'surprise': 'surprised',  // 惊讶 → 张嘴
    'neutral': 'neutral'      // 中性 → 平静
};
```

### 测试表情
```javascript
// 开心 90%
ani.setExpression('joy', 0.9)

// 难过 70%
ani.setExpression('sad', 0.7)

// 惊讶 100%
ani.setExpression('surprise', 1.0)

// 回到中性
ani.setExpression('neutral', 1.0)
```

## 🚀 性能优化

### 当前性能目标
- 60 FPS 渲染
- <2s LLM响应（Mock模式）
- <4s TTS生成
- 实时动画更新

### 如果 FPS 过低
1. 减少动画计算频率
2. 简化手势系统
3. 关闭部分微动作

### 如果需要更流畅
已经优化：
- 使用 Three.Clock 精确计时
- 分层动画系统
- 高效的骨骼查找缓存

## 📝 下一步增强

### 可选增强功能
1. **眼球追踪** - 让眼睛跟随鼠标
2. **眨眼动画** - 随机眨眼
3. **情绪手势映射** - joy时挥手，sad时低头
4. **物理效果** - 头发/裙子摆动
5. **高级唇形** - 基于音素的精确唇形
6. **表情预设** - 保存和加载姿势
7. **录制功能** - 录制动画序列

## 🎯 快速问题排查

### 问题：角色不动
**检查：**
```javascript
// Console 中检查
ani.vrm()  // 应该返回 VRM 对象
ani.vrm().humanoid  // 应该有 humanoid
```

### 问题：手臂姿势不对
**解决：**
1. 打开 http://localhost:8000/pose
2. 调整滑块
3. 点击 "Copy Values"
4. 更新代码

### 问题：动画太夸张/太微妙
**解决：**
参考上方"动画强度调整"部分修改数值

### 问题：说话没有手势
**检查：**
```javascript
// 确保 talkingAnimation 不为 null
console.log(talkingAnimation)
```

## 💡 最佳实践

1. **先用姿势工具调试** - 用 /pose 页面找到完美姿势
2. **逐步测试** - 先测试空闲，再测试说话，最后测试手势
3. **保存好的值** - 记录满意的旋转角度
4. **适度动画** - 不要让所有动画都太夸张
5. **测试不同情绪** - 确保每个表情都自然

---

**当前状态：**
✅ T-pose 已修复
✅ 空闲动画已实现（呼吸、摇摆、头部、手臂、手指）
✅ 手势系统完整（6种手势）
✅ 说话动画已集成
✅ 表情系统已映射
✅ 中文语音回复已支持

**下一步建议：**
使用 http://localhost:8000/pose 微调手臂到完美姿势！
