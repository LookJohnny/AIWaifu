# VRMA 专业动画设置指南

## 🎯 目标
使用 VRoid Hub 提供的专业动作捕捉动画文件，让 Ani 的动作完全自然。

---

## 📥 步骤 1：下载免费 VRMA 文件

### 方法 A：从 BOOTH 下载（官方推荐）

1. **访问 BOOTH 页面：**
   ```
   https://vroid.booth.pm/items/5512385
   ```

2. **免费获取：**
   - 点击 "Free Download"（免费下载）
   - 可能需要注册 BOOTH 账号（免费）
   - 下载 ZIP 文件

3. **解压文件：**
   - 解压到任意位置
   - 你会得到 7 个 .vrma 文件：
     - VRMA_01_ShowFullBody.vrma
     - VRMA_02_Greeting.vrma
     - VRMA_03_PeaceSign.vrma
     - VRMA_04_Shoot.vrma
     - VRMA_05_Spin.vrma
     - VRMA_06_ModelPose.vrma
     - VRMA_07_Squat.vrma

4. **复制到项目：**
   ```
   复制所有 .vrma 文件到：F:\Ani\animations\
   ```

### 方法 B：找到合适的 Idle 动画（推荐用于空闲状态）

如果你想要更自然的 idle（空闲）动画，可以从以下来源获取：

1. **BOOTH 市场搜索：**
   - 访问：https://booth.pm
   - 搜索："VRM Animation idle" 或 "VRMA idle"
   - 很多免费的 idle 动画

2. **推荐免费资源：**
   - VRoid Hub Photo Booth 动画
   - BOOTH 上标记为 "Free" 的 VRMA

---

## 🔧 步骤 2：已为你准备好集成代码

我已经创建了完整的 VRMA 加载系统：

### 文件位置：
```
F:\Ani\frontend\avatar_vrma.html
```

### 功能特性：
- ✅ 自动加载 VRMA 文件
- ✅ 循环播放空闲动画
- ✅ 平滑过渡
- ✅ 与表情系统集成
- ✅ 保持说话时的动画

---

## 🎮 步骤 3：使用 VRMA 系统

### 下载完成后：

1. **确保文件位置正确：**
   ```
   F:\Ani\animations\VRMA_02_Greeting.vrma  （或其他 .vrma 文件）
   ```

2. **打开 VRMA 版本：**
   ```
   http://localhost:8000/vrma
   ```

3. **测试效果：**
   - 角色应该使用专业动画
   - 手臂姿势完美
   - 动作自然流畅

---

## 🎨 自定义动画选择

### 修改使用哪个动画

编辑 `frontend/avatar_vrma.html` 找到：

```javascript
// 加载 VRMA 动画
const animationPath = '/animations/VRMA_02_Greeting.vrma';  // ← 修改这里
```

### 推荐的动画用途：

| 动画文件 | 用途 | 说明 |
|---------|------|------|
| VRMA_02_Greeting.vrma | 默认空闲 | 打招呼姿势，友好自然 |
| VRMA_06_ModelPose.vrma | 展示姿势 | 标准站姿，手臂自然 |
| VRMA_03_PeaceSign.vrma | 开心状态 | 比耶手势，活泼可爱 |
| VRMA_01_ShowFullBody.vrma | 展示全身 | T-pose变体，展示用 |

---

## 🔄 步骤 4：情绪-动画映射（高级）

如果你想根据情绪自动切换动画：

```javascript
const emotionAnimations = {
    'joy': '/animations/VRMA_03_PeaceSign.vrma',      // 开心 → 比耶
    'neutral': '/animations/VRMA_02_Greeting.vrma',   // 中性 → 打招呼
    'surprise': '/animations/VRMA_04_Shoot.vrma',     // 惊讶 → 射击动作
    'sad': '/animations/VRMA_06_ModelPose.vrma'       // 难过 → 安静站立
};

function loadAnimationForEmotion(emotion) {
    const animPath = emotionAnimations[emotion] || emotionAnimations['neutral'];
    loadVRMAAnimation(animPath);
}
```

---

## 📊 预期效果对比

### 之前（手动动画）：
- ⚠️ 手臂可能不自然
- ⚠️ 动作可能僵硬
- ⚠️ 需要手动调试角度

### 之后（VRMA专业动画）：
- ✅ 完美的手臂姿势
- ✅ 流畅的动作
- ✅ 专业级质量
- ✅ 开箱即用

---

## 🐛 常见问题

### Q: 动画文件加载失败？
**A:** 检查：
1. 文件路径是否正确：`F:\Ani\animations\xxx.vrma`
2. 文件名是否匹配代码中的路径
3. 浏览器控制台（F12）是否有错误

### Q: 动画播放不流畅？
**A:** VRMA 使用关键帧插值，应该非常流畅。如果不流畅：
1. 检查 FPS（应该60fps）
2. 关闭其他耗资源的程序
3. 查看控制台错误

### Q: 想要更多动画？
**A:**
1. 访问 BOOTH.pm 搜索 "VRMA"
2. 很多创作者提供免费/付费动画
3. 也可以自己用 VRM Posing Desktop 制作

### Q: 能否混合多个动画？
**A:** 可以！使用 AnimationMixer 的 `fadeIn/fadeOut`：
```javascript
// 从当前动画淡出，新动画淡入
currentAction.fadeOut(0.5);
newAction.reset().fadeIn(0.5).play();
```

---

## 🚀 立即开始

### 最简单的流程：

1. **下载文件（5分钟）：**
   - 访问：https://vroid.booth.pm/items/5512385
   - 免费下载
   - 解压到 F:\Ani\animations\

2. **打开 VRMA 版本（1秒）：**
   ```
   http://localhost:8000/vrma
   ```

3. **享受专业动画！**

---

## 📝 技术细节

### VRMA 格式优势：
- 基于 glTF 2.0
- 支持关键帧动画
- 包含表情和姿势数据
- 标准化的人形骨骼映射
- 跨应用兼容

### 加载流程：
1. GLTFLoader 加载 .vrma 文件
2. VRMAnimationLoaderPlugin 解析动画数据
3. 创建 AnimationClip
4. AnimationMixer 播放
5. 自动循环

### 性能：
- 文件大小：通常 10-50KB
- 加载时间：<1秒
- 运行开销：极低
- FPS 影响：几乎无

---

## 💡 下一步建议

完成 VRMA 集成后，你可以：

1. **收集更多动画：**
   - BOOTH 上搜索更多 VRMA
   - 建立自己的动画库
   - 不同场景使用不同动画

2. **实现动画切换：**
   - 根据对话内容切换
   - 根据情绪自动选择
   - 时间触发（每30秒换一个）

3. **自制动画：**
   - 使用 VRM Posing Desktop
   - 导出为 VRMA
   - 完全定制化

---

## ✅ 检查清单

准备使用 VRMA 前，确保：

- [ ] 已下载 VRMA 文件
- [ ] 文件在 F:\Ani\animations\ 目录
- [ ] 服务器正在运行
- [ ] 浏览器支持 WebGL
- [ ] 已测试 /vrma 页面加载

全部完成后，你就有了专业级的 3D 角色动画系统！🎉
