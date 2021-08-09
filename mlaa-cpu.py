import argparse
import os
from tqdm import tqdm
import numpy as np

if (__package__ == '') or (__package__ is None):
    import sys
    sys.path.append(os.path.abspath(
        os.path.dirname(os.path.dirname(__file__))))
    import utils

else:
    import utils


def _get_luminance_img(img):
    l_img = 0.2126*img[:, :, 0]+0.7152*img[:, :, 1]+0.0722*img[:, :, 2]
    return l_img


def _find_edges(img, th=0.1):
    buffer = np.zeros((img.shape[0], img.shape[1], 3))
    for y in range(1, img.shape[0]):
        for x in range(0, img.shape[1]):
            if abs(img[y, x]-img[y-1, x]) > th:
                buffer[y, x, 1] = 1
    for y in range(0, img.shape[0]):
        for x in range(1, img.shape[1]):
            if abs(img[y, x]-img[y, x-1]) > th:
                buffer[y, x, 0] = 1
    return buffer


def _cal_aliasing_info_x(img_edges, start_x, start_y, mask):
    dis = 1
    for x in range(start_x, img_edges.shape[1]):
        if img_edges[start_y, x, 0] == 1 and img_edges[start_y-1, x, 0] == 1:
            pattern = 'H'
            return dis, pattern, mask
        if img_edges[start_y, x, 0] == 1:
            pattern = 'T'
            return dis, pattern, mask
        if img_edges[start_y-1, x, 0] == 1:
            pattern = 'B'
            return dis, pattern, mask
        if img_edges[start_y, x, 1] == 0:
            break
        mask[start_y, x] = 1
        dis+=1
    pattern = 'L'
    return dis, pattern, mask


def _cal_aliasing_info_y(img_edges, start_x, start_y, mask):
    dis = 1
    for y in range(start_y, img_edges.shape[0]):
        if img_edges[y, start_x, 1] == 1 and img_edges[y, start_x-1, 1] == 1:
            pattern = 'H'
            return dis, pattern, mask
        if img_edges[y, start_x, 1] == 1:
            pattern = 'T'
            return dis, pattern, mask
        if img_edges[y, start_x-1, 1] == 1:
            pattern = 'B'
            return dis, pattern, mask
        if img_edges[y, start_x, 0] == 0:
            break
        mask[y, start_x] = 1
        dis+=1
    pattern = 'L'
    return dis, pattern, mask


def _find_aliasings_x(img_edges):
    list_aliasings = []
    mask = np.zeros((img_edges.shape[0], img_edges.shape[1], 1))
    for y in range(1, img_edges.shape[0]):
        for x in range(0, img_edges.shape[1]):
            if mask[y, x] == 0:
                if img_edges[y, x, 1] == 1:
                    if img_edges[y, x, 0] == 1 and img_edges[y-1, x, 0] == 1:
                        start_pattern = 'H'
                    elif img_edges[y, x, 0] == 1:
                        start_pattern = 'T'
                    elif img_edges[y-1, x, 0] == 1:
                        start_pattern = 'B'
                    else:
                        start_pattern = 'L'
                    dis, end_pattern, mask = _cal_aliasing_info_x(
                            img_edges, x+1, y, mask)
                    list_aliasings.append(
                        [y, x, dis, start_pattern+end_pattern])
    return list_aliasings


def _find_aliasings_y(img_edges):
    list_aliasings = []
    mask = np.zeros((img_edges.shape[0], img_edges.shape[1], 1))
    for x in range(1, img_edges.shape[1]):
        for y in range(0, img_edges.shape[0]):
            if mask[y, x] == 0:
                if img_edges[y, x, 0] == 1:
                    if img_edges[y, x, 1] == 1 and img_edges[y, x-1, 1] == 1:
                        start_pattern = 'H'
                    elif img_edges[y, x, 1] == 1:
                        start_pattern = 'T'
                    elif img_edges[y, x-1, 1] == 1:
                        start_pattern = 'B'
                    else:
                        start_pattern = 'L'
                    dis, end_pattern, mask = _cal_aliasing_info_y(
                            img_edges, x, y+1, mask)
                    list_aliasings.append(
                        [y, x, dis, start_pattern+end_pattern])
    return list_aliasings


def _analyse_pattern(pattern):
    if pattern[0] == 'H':
        if pattern[1] == 'H':
            start = 0
            end = 0
        elif pattern[1] == 'T':
            start = -0.5
            end = 0.5
        elif pattern[1] == 'B':
            start = 0.5
            end = -0.5
        elif pattern[1] == 'L':
            start = 0
            end = 0
    elif pattern[0] == 'T':
        if pattern[1] == 'H':
            start = 0.5
            end = -0.5
        elif pattern[1] == 'T':
            start = 0.5
            end = 0.5
        elif pattern[1] == 'B':
            start = 0.5
            end = -0.5
        elif pattern[1] == 'L':
            start = 0.5
            end = 0
    elif pattern[0] == 'B':
        if pattern[1] == 'H':
            start = -0.5
            end = 0.5
        elif pattern[1] == 'T':
            start = -0.5
            end = 0.5
        elif pattern[1] == 'B':
            start = -0.5
            end = -0.5
        elif pattern[1] == 'L':
            start = -0.5
            end = 0
    elif pattern[0] == 'L':
        if pattern[1] == 'H':
            start = 0
            end = 0
        elif pattern[1] == 'T':
            start = 0
            end = 0.5
        elif pattern[1] == 'B':
            start = 0
            end = -0.5
        elif pattern[1] == 'L':
            start = 0
            end = 0
    return start, end


def _cal_area_list(dis, pattern):
    start, end = _analyse_pattern(pattern)

    if start == 0 and end == 0:
        return None
    elif end == 0:
        h = start
        tri_len = dis
    elif start == 0:
        h = end
        tri_len = dis
    else:
        h = start
        tri_len = dis/2.0

    list_area = np.zeros((dis, 2))
    tri_area = abs(h)*tri_len/2

    if start==0:
        for i in range(0, dis):
            area = (end*2)*(tri_area*(((i+1)/tri_len)**2) -
                            tri_area*(((i)/tri_len)**2))
            if area > 0:
                list_area[i, 0] = area
            else:
                list_area[i, 1] = -area
    elif end==0:
        for i in range(0, dis):
            area = (start*2)*(tri_area*(((tri_len-i)/tri_len)
                                        ** 2)-tri_area*(((tri_len-i-1)/tri_len)**2))
            if area > 0:
                list_area[i, 0] = area
            else:
                list_area[i, 1] = -area
    elif tri_len % 2 == 0:
        for i in range(0, dis+1):
            if i == tri_len:
                continue
            elif i < tri_len:
                area = (start*2)*(tri_area*(((tri_len-i)/tri_len)
                                            ** 2)-tri_area*(((tri_len-i-1)/tri_len)**2))
                if area > 0:
                    list_area[i, 0] = area
                else:
                    list_area[i, 1] = -area
            elif i > tri_len:
                area = (end*2)*(tri_area*(((i-tri_len)/tri_len)**2) -
                                tri_area*(((i-tri_len-1)/tri_len)**2))
                if area > 0:
                    list_area[i-1, 0] = area
                else:
                    list_area[i-1, 1] = -area
    else:
        for i in range(0, dis+1):
            if abs(i-tri_len) <= 0.5:
                if i < tri_len:
                    area = (start*2)*(tri_area*(((tri_len-i)/tri_len)**2))
                    if area > 0:
                        list_area[i, 0] += area
                    else:
                        list_area[i, 1] -= area
                else:
                    area = (end*2)*(tri_area*(((i-tri_len)/tri_len)**2))
                    if area > 0:
                        list_area[i-1, 0] += area
                    else:
                        list_area[i-1, 1] -= area
            elif i < tri_len:
                area = (start*2)*(tri_area*(((tri_len-i)/tri_len)
                                            ** 2)-tri_area*(((tri_len-i-1)/tri_len)**2))
                if area > 0:
                    list_area[i, 0] = area
                else:
                    list_area[i, 1] = -area
            elif i > tri_len:
                area = (end*2)*(tri_area*(((i-tri_len)/tri_len)**2) -
                                tri_area*(((i-tri_len-1)/tri_len)**2))
                if area > 0:
                    list_area[i-1, 0] = area
                else:
                    list_area[i-1, 1] = -area
    return list_area


def _update_weights_x(weights, list_area, start_y, start_x):
    for x in range(start_x, start_x+len(list_area)):
        weights[start_y, x, 0] = list_area[x-start_x, 0]
        weights[start_y, x, 1] = list_area[x-start_x, 1]
    return weights


def _update_weights_y(weights, list_area, start_y, start_x):
    for y in range(start_y, start_y+len(list_area)):
        weights[y, start_x, 2] = list_area[y-start_y, 0]
        weights[y, start_x, 3] = list_area[y-start_y, 1]
    return weights


def _get_weights(img_shape, list_aliasing_x, list_aliasing_y):
    weights = np.zeros((img_shape[0], img_shape[1], 4))
    for [start_y, start_x, dis, pattern] in list_aliasing_x:
        list_area = _cal_area_list(dis, pattern)
        if list_area is None:
            continue
        weights = _update_weights_x(
            weights, list_area, start_y, start_x)
    for [start_y, start_x, dis, pattern] in list_aliasing_y:
        list_area = _cal_area_list(dis, pattern)
        if list_area is None:
            continue
        weights = _update_weights_y(
            weights, list_area, start_y, start_x)
    return weights


def _blend_color(img_in, img_weight):
    img_blended= np.zeros((img_in.shape[0],img_in.shape[1]))
    for y in range(0, img_in.shape[0]):
        for x in range(0, img_in.shape[1]):
            img_blended[y, x]=(2-img_weight[y,x,0]-img_weight[y,x,2])*img_in[y,x]
            if y!=0:
                img_blended[y, x]+=img_in[y-1,x]*img_weight[y,x,0]
            if y!=img_in.shape[0]-1:
                img_blended[y, x]+=(img_in[y+1,x]-img_in[y,x])*img_weight[y+1,x,1]
            if x!=0:
                img_blended[y, x]+=img_in[y,x-1]*img_weight[y,x,2]
            if x!=img_in.shape[1]-1:
                img_blended[y, x]+=(img_in[y,x+1]-img_in[y,x])*img_weight[y,x+1,3]
            img_blended[y, x]/=2
    return img_blended


def mlaa_img_luminance(img, num_th):
    img_lu = _get_luminance_img(img)

    utils.save_img_float('halfway/luminance.bmp', img_lu)

    img_edge = _find_edges(img_lu, num_th)

    utils.save_img_float('halfway/edge.bmp', img_edge)

    list_aliasing_x = _find_aliasings_x(img_edge)
    list_aliasing_y = _find_aliasings_y(img_edge)

    img_weight = _get_weights(img_lu.shape, list_aliasing_x, list_aliasing_y)
    
    utils.save_img_float('halfway/weights.bmp', img_weight)

    img[:,:,0] = _blend_color(img[:,:,0], img_weight)
    img[:,:,1] = _blend_color(img[:,:,1], img_weight)
    img[:,:,2] = _blend_color(img[:,:,2], img_weight)
    return img


def mlaa_img_perchannel(img, num_th):
    for i in range(0,3):
        img_edge = _find_edges(img[:,:,i], num_th)

        list_aliasing_x = _find_aliasings_x(img_edge)
        list_aliasing_y = _find_aliasings_y(img_edge)

        img_weight = _get_weights(img[:,:,i].shape, list_aliasing_x, list_aliasing_y)
        
        img[:,:,i] = _blend_color(img[:,:,i], img_weight)
    return img


def mlaa_imgs(dir_ori, dir_tar, str_suffix,str_type, num_th):
    list_files = os.listdir(dir_ori)
    num_files = len(list_files)
    with tqdm(total=num_files, desc='MLAA') as bar:
        for file in list_files:
            img = utils.load_img_float(os.path.join(dir_ori, file))

            if str_type=='L':
                img = mlaa_img_luminance(img, num_th)
            elif str_type=='P':
                img = mlaa_img_perchannel(img, num_th)
            else:
                raise ValueError("Edge finding type not found: "+str_type+'. (Default: L)')

            utils.save_img_float(os.path.join(
                dir_tar, file[:-4]+str_suffix), img)

            bar.update(1)


def main(dir_ori, dir_tar, str_suffix,str_type, num_th):
    dir_work = os.path.abspath(os.path.dirname(__file__))
    print('Running in: '+dir_work)
    utils.check_folder(dir_ori)
    if dir_tar == '.':
        dir_tar = dir_ori+'_out'
    utils.init_folder(dir_tar)

    mlaa_imgs(dir_ori, dir_tar, str_suffix,str_type, num_th)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Using morphological antialiasing to process imgs')
    parser.add_argument('--origin', dest='dir_ori', type=str,
                        help='folder of origin images', required=True)
    parser.add_argument('--target', dest='dir_tar',
                        type=str, default='.', help='target folder of result images. Default path is "{DIR_ORI}_out/"', required=False)
    parser.add_argument('--save', dest='str_suffix',
                        type=str, help='result images suffix', default='.bmp', required=False)
    parser.add_argument('--type', dest='str_type',
                        type=str, help='type string for finding image edges. "L": using luminance; "P": using per-channel color value', default='L', required=False)
    parser.add_argument('--th', dest='num_th',
                        type=float, help='threshold for finding edges', default=0.1, required=False)
    args = parser.parse_args()
    main(args.dir_ori, args.dir_tar, args.str_suffix, args.str_type, args.num_th)
