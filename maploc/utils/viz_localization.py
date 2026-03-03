# Copyright (c) Meta Platforms, Inc. and affiliates.

import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch


def likelihood_overlay(
    prob, map_viz=None, p_rgb=0.2, p_alpha=1 / 15, thresh=None, cmap="jet"
):
    prob = prob / prob.max()
    cmap = plt.get_cmap(cmap)
    rgb = cmap(prob**p_rgb)
    alpha = prob[..., None] ** p_alpha
    if thresh is not None:
        alpha[prob <= thresh] = 0
    if map_viz is not None:
        faded = map_viz + (1 - map_viz) * 0.5
        rgb = rgb[..., :3] * alpha + faded * (1 - alpha)
        rgb = np.clip(rgb, 0, 1)
    else:
        rgb[..., -1] = alpha.squeeze(-1)
    return rgb


def heatmap2rgb(scores, mask=None, clip_min=0.05, alpha=0.8, cmap="jet"):
    min_, max_ = np.quantile(scores, [clip_min, 1])
    scores = scores.clip(min=min_)
    rgb = plt.get_cmap(cmap)((scores - min_) / (max_ - min_))
    if mask is not None:
        if alpha == 0:
            rgb[mask] = np.nan
        else:
            rgb[..., -1] = 1 - (1 - 1.0 * mask) * (1 - alpha)
    return rgb


def plot_pose(axs, xy, yaw=None, s=1 / 25, c="r", a=1, w=0.015, dot=True, zorder=10):
    if yaw is not None:
        yaw = np.deg2rad(yaw)
        uv = np.array([np.sin(yaw), -np.cos(yaw)])
    xy = np.array(xy) + 0.5
    if not isinstance(axs, list):
        axs = [axs]
    for ax in axs:
        if isinstance(ax, int):
            ax = plt.gcf().axes[ax]
        if dot:
            ax.scatter(*xy, c=c, s=30, zorder=zorder, linewidths=0, alpha=a)
        if yaw is not None:
            ax.quiver(
                *xy,
                *uv,
                scale=s,
                scale_units="xy",
                angles="xy",
                color=c,
                zorder=zorder,
                alpha=a,
                width=w,
            )


def plot_dense_rotations(
    ax, prob, thresh=0.0001, skip=10, s=1 / 15, k=3, c="k", w=None, **kwargs
):
    t = torch.argmax(prob, -1)
    yaws = t.numpy() / prob.shape[-1] * 360
    prob = prob.max(-1).values / prob.max()
    mask = prob > thresh
    masked = prob.masked_fill(~mask, 0)
    max_ = torch.nn.functional.max_pool2d(
        masked.float()[None, None], k, stride=1, padding=k // 2
    )
    mask = (max_[0, 0] == masked.float()) & mask
    indices = np.where(mask.numpy() > 0)

    # 打印箭头位置和朝向（注意 indices[::-1] 是 (x, y)）
    xs, ys = indices[1], indices[0]
    yaws_selected = yaws[ys, xs]
    for x, y, yaw in zip(xs, ys, yaws_selected):
        print(f"Arrow at (x={x}, y={y}), yaw={yaw:.2f} degrees")

    plot_pose(
        ax,
        indices[::-1],
        yaws[indices],
        s=s,
        c=c,
        dot=False,
        zorder=0.1,
        w=w,
        **kwargs,
    )
    return indices,yaws_selected,yaws


def copy_image(im, ax):
    prop = im.properties()
    prop.pop("children")
    prop.pop("size")
    prop.pop("tightbbox")
    prop.pop("transformed_clip_path_and_affine")
    prop.pop("window_extent")
    prop.pop("figure")
    prop.pop("transform")
    prop.pop("shape", None)
    return ax.imshow(im.get_array(), **prop)


def add_circle_inset(
    ax,#对象，代表在其上绘制圆形插图的 Matplotlib 画布
    center,#圆形的中心坐标，在原始 ax 上的坐标。
    ax1=None,
    corner=None,#插图的角落位置，用于确定插图的位置。如果未指定，默认会计算插图的位置。
    radius_px=10,#圆的半径，以像素为单位。
    inset_size=0.4,#插图的大小比例，值越大，插图占 ax 的区域越大。
    inset_offset=0.005,#插图的位置偏移量，防止插图完全对齐原始画布的边缘。
    color="red",#圆形的颜色。
):
    data_t_axes = ax.transAxes + ax.transData.inverted() #两者相加是为了完成从数据坐标到 Axes 坐标的双重转换，准备后续使用。
    if corner is None:
        center_axes = np.array(data_t_axes.inverted().transform(center))
        corner = 1 - np.round(center_axes).astype(int)
        corner = [0, 0]  # 右下角
    corner = np.array(corner)#转换为 NumPy 数组，以便于进行数学运算。
    bottom_left = corner * (1 - inset_size - inset_offset) + (1 - corner) * inset_offset #根据插图的大小比例和偏移量来调整左下角位置。
    axins = ax.inset_axes([*bottom_left, inset_size, inset_size])#用于在 ax 上创建一个插图区域。bottom_left 是左下角位置，inset_size 控制插图区域的大小。
    if ax.yaxis_inverted():#如果原始 ax 的 y 轴是反转的，那么插图区域也要反转 y 轴。
        axins.invert_yaxis()
    axins.set_axis_off()#关闭插图区域的坐标轴显示，使插图更干净。

    c = mpl.patches.Circle(center, radius_px, fill=False, color=color)#创建一个圆形补丁 mpl.patches.Circle，圆心在 center，半径为 radius_px，颜色为 color。
    ax.add_patch(copy.deepcopy(c))#将该圆形补丁添加到原始 ax 和插图 axins 中。
    axins.add_patch(c)#使用 copy.deepcopy(c) 是为了避免引用问题，确保 c 在 ax 和 axins 上分别是独立的。

    radius_inset = radius_px + 1
    axins.set_xlim([center[0] - radius_inset, center[0] + radius_inset])
    ylim = center[1] - radius_inset, center[1] + radius_inset
    if axins.yaxis_inverted():
        ylim = ylim[::-1]
    axins.set_ylim(ylim)

    for im in ax.images:
        im2 = copy_image(im, axins)
        im2.set_clip_path(c)

    # 如果提供了ax1，将局部放大图绘制到ax1中同样的位置
    if ax1 is not None:
        # 在ax1中创建相同位置的inset
        ax1_inset = ax1.inset_axes([*bottom_left, inset_size, inset_size])
        if ax1.yaxis_inverted():
            ax1_inset.invert_yaxis()
        ax1_inset.set_axis_off()
        
        # 为ax1创建独立的圆圈对象
        #c_ax1 = mpl.patches.Circle(center, radius_px, fill=False, color=color)
        c_ax1_inset = mpl.patches.Circle(center, radius_px, fill=False, color=color)
        #ax1.add_patch(c_ax1)
        ax1_inset.add_patch(c_ax1_inset)
        
        # 设置相同的坐标范围
        ax1_inset.set_xlim([center[0] - radius_inset, center[0] + radius_inset])
        ylim_ax1 = center[1] - radius_inset, center[1] + radius_inset
        if ax1_inset.yaxis_inverted():
            ylim_ax1 = ylim_ax1[::-1]
        ax1_inset.set_ylim(ylim_ax1)
        
        # 将ax中的图像复制到ax1的inset中
        for im in ax.images:
            im2_ax1 = copy_image(im, ax1_inset)
            im2_ax1.set_clip_path(c_ax1_inset)
    
    return axins


def plot_bev(bev, uv, yaw, ax=None, zorder=10, **kwargs):
    if ax is None:
        ax = plt.gca()
    h, w = bev.shape[:2]
    tfm = mpl.transforms.Affine2D().translate(-w / 2, -h)
    tfm = tfm.rotate_deg(yaw).translate(*uv + 0.5)
    tfm += plt.gca().transData
    ax.imshow(bev, transform=tfm, zorder=zorder, **kwargs)
    ax.plot(
        [0, w - 1, w / 2, 0],
        [0, 0, h - 0.5, 0],
        transform=tfm,
        c="k",
        lw=1,
        zorder=zorder + 1,
    )


def plot_logistics_lines(
        ax,
        start_point,
        prob,
        thresh=0.01,
        max_lines=50,
        color="red",
        alpha_range=(0.3, 1.0),
        linewidth_range=(0.5, 2.0),
        zorder=5,
        special_point=None,
        special_color="red"
):
    """
    从起始点发射线条到概率分布中的高概率位置，类似物流网络可视化

    Args:
        ax: matplotlib轴对象
        start_point: 起始点坐标 (x, y)
        prob: 概率分布数组，形状为 (H, W) 或 (H, W, rotations)
        thresh: 概率阈值，只绘制高于此值的位置
        max_lines: 最大线条数量
        color: 默认线条颜色
        alpha_range: 透明度范围 (min_alpha, max_alpha)
        linewidth_range: 线宽范围 (min_width, max_width)
        zorder: 绘制层级
        special_point: 特殊点坐标 (x, y)，如果指定，到此点的连线将使用特殊颜色
        special_color: 特殊点连线的颜色
    """
    print(f"plot_logistics_lines called with prob shape: {prob.shape}, thresh: {thresh}")

    # 参考plot_dense_rotations的处理方式来提取位置和概率
    # 如果有旋转维度，取最大概率值
    if len(prob.shape) == 3:
        prob_2d = prob.max(-1).values / prob.max()
    else:
        prob_2d = prob / prob.max()

    print(f"prob_2d shape: {prob_2d.shape}, max: {prob_2d.max():.4f}")

    # 创建阈值掩码
    mask = prob_2d > thresh
    print(f"Points above threshold {thresh}: {mask.sum()}")

    # 如果没有足够的点，降低阈值
    if mask.sum() < 5:
        thresh = thresh / 2
        mask = prob_2d > thresh
        print(f"Lowered threshold to {thresh}, points found: {mask.sum()}")

    # 使用最大池化找到局部最大值，避免过于密集的线条
    k = 3  # 池化核大小
    masked = prob_2d.masked_fill(~mask, 0)
    max_pooled = torch.nn.functional.max_pool2d(
        masked.float()[None, None], k, stride=1, padding=k // 2
    )
    local_max_mask = (max_pooled[0, 0] == masked.float()) & mask

    print(f"After max pooling: {local_max_mask.sum()} local maxima found")

    # 获取满足条件的位置索引
    indices = np.where(local_max_mask.numpy() > 0)

    if len(indices[0]) == 0:
        print("No points found even with lowered threshold")
        return

    print(f"Found {len(indices[0])} candidate points")

    # 获取对应的概率值
    prob_values = prob_2d[indices].numpy()

    # 按概率值排序，优先绘制高概率的线条
    sorted_idx = np.argsort(prob_values)[::-1]  # 降序排列

    # 限制线条数量
    n_lines = min(max_lines, len(sorted_idx))

    # 智能选择线条，避免过度集中和交叉
    selected_positions = []
    selected_idx = []
    filtered_count = 0

    for idx in sorted_idx:
        if len(selected_idx) >= n_lines:
            break

        y_pos, x_pos = indices[0][idx], indices[1][idx]
        current_pos = np.array([x_pos, y_pos])

        # 检查与已选择位置的距离，避免过于密集
        too_close = False
        min_distance = 5  # 最小距离阈值（降低以允许更多线条）

        for selected_pos in selected_positions:
            if np.linalg.norm(current_pos - selected_pos) < min_distance:
                too_close = True
                filtered_count += 1
                break

        if not too_close:
            selected_positions.append(current_pos)
            selected_idx.append(idx)

    print(f"Filtered out {filtered_count} points due to proximity")

    # 按角度排序选中的点，进一步优化视觉效果
    if len(selected_positions) > 1:
        start_pos = np.array(start_point)
        angles = []
        for pos in selected_positions:
            dx = pos[0] - start_pos[0]
            dy = pos[1] - start_pos[1]
            angle = np.arctan2(dy, dx)
            angles.append(angle)

        # 按角度排序
        angle_sorted_idx = np.argsort(angles)
        selected_idx = [selected_idx[i] for i in angle_sorted_idx]

    n_lines = len(selected_idx)
    print(f"Will draw {n_lines} lines")

    # 起始点坐标（添加0.5偏移，与plot_pose保持一致）
    start_x, start_y = np.array(start_point) + 0.5
    print(f"Start point: ({start_x:.1f}, {start_y:.1f})")

    # 绘制线条
    for i, idx in enumerate(selected_idx):
        y_idx, x_idx = indices[0][idx], indices[1][idx]
        prob_val = prob_values[idx]

        # 目标点坐标（添加0.5偏移）
        end_x, end_y = x_idx + 0.5, y_idx + 0.5

        # 检查是否是特殊点
        is_special = False
        current_color = color
        if special_point is not None:
            # 检查当前点是否接近特殊点（允许一定的误差）
            special_x, special_y = special_point
            distance_to_special = np.sqrt((end_x - special_x - 0.5) ** 2 + (end_y - special_y - 0.5) ** 2)
            if distance_to_special < 15:  # 3像素的容差
                is_special = True
                current_color = special_color
                print(f"Found special point at ({end_x:.1f}, {end_y:.1f}), using color {special_color}")

        # 根据概率值计算视觉属性
        # 使用概率值的相对排名来计算透明度和线宽
        if n_lines > 1:
            rank_norm = i / (n_lines - 1)  # 从0到1，0是最高概率
        else:
            rank_norm = 0

        # 反转rank_norm，让高概率的线条更明显
        prob_norm = 1.0 - rank_norm
        alpha = alpha_range[0] + prob_norm * (alpha_range[1] - alpha_range[0])
        linewidth = linewidth_range[0] + prob_norm * (linewidth_range[1] - linewidth_range[0])

        # 如果是特殊点，使用非常鲜艳显眼的效果
        if is_special:
            alpha = 1.0  # 完全不透明
            linewidth = linewidth * 2.5  # 大幅增加线宽，更显眼
            current_color = special_color  # 确保使用特殊颜色

        # 计算弧形路径，增加立体感
        # 计算距离和方向
        dx = end_x - start_x
        dy = end_y - start_y
        distance = np.sqrt(dx ** 2 + dy ** 2)

        # 根据距离和概率计算弧度高度
        # 所有弧度方向保持一致
        arc_height = distance * 0.3 * (1 + 0.5 * prob_norm)  # 高概率的弧度更大
        direction = 1 if i % 2 == 0 else -1  # 交替方向，避免单调

        # 计算中点和控制点
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2

        # 计算垂直于连线的方向向量
        if distance > 0:
            perp_x = -dy / distance * arc_height * direction
            perp_y = dx / distance * arc_height * direction
        else:
            perp_x = perp_y = 0

        # 控制点位置
        ctrl_x = mid_x + perp_x
        ctrl_y = mid_y + perp_y

        # 生成贝塞尔曲线点
        t = np.linspace(0, 1, 50)
        # 二次贝塞尔曲线公式
        curve_x = (1 - t) ** 2 * start_x + 2 * (1 - t) * t * ctrl_x + t ** 2 * end_x
        curve_y = (1 - t) ** 2 * start_y + 2 * (1 - t) * t * ctrl_y + t ** 2 * end_y

        # 绘制弧形线条
        ax.plot(
            curve_x,
            curve_y,
            color=current_color,
            alpha=alpha,
            linewidth=linewidth,
            zorder=zorder,
            solid_capstyle='round'
        )

        # 在终点添加小箭头指示方向
        if distance > 5:  # 只在较长的线条上添加箭头
            # 计算箭头方向（曲线末端的切线方向）
            arrow_dx = curve_x[-1] - curve_x[-5]
            arrow_dy = curve_y[-1] - curve_y[-5]
            arrow_length = np.sqrt(arrow_dx ** 2 + arrow_dy ** 2)
            if arrow_length > 0:
                arrow_dx /= arrow_length
                arrow_dy /= arrow_length

                # 绘制小箭头
                arrow_alpha = 1.0 if is_special else alpha * 0.8  # 特殊点箭头完全不透明
                arrow_lw = linewidth if is_special else linewidth * 0.8  # 特殊点箭头更粗

                ax.annotate('',
                            xy=(end_x, end_y),
                            xytext=(end_x - arrow_dx * 3, end_y - arrow_dy * 3),
                            arrowprops=dict(arrowstyle='-|>',
                                            color=current_color,
                                            alpha=arrow_alpha,
                                            lw=arrow_lw,
                                            shrinkA=0,
                                            shrinkB=0))
        if i < 5:  # 只打印前5条线的信息
            print(
                f"Line {i}: ({start_x:.1f},{start_y:.1f}) -> ({end_x:.1f},{end_y:.1f}), prob: {prob_val:.4f}, alpha: {alpha:.2f}")

    print(f"Logistics lines drawn successfully!")
