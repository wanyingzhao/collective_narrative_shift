"""
This script provide func to map Propbank frame 
to Verbatlas frame

Will return None when there is no match
"""
import pandas as pd

######## load map file ########
MAP_FPATH = "/home/zhaowany/INCAS/incas-iu/data/derived/pb2va/prop2va.csv"
INFO_FPATH = "/home/zhaowany/INCAS/incas-iu/data/derived/pb2va/VA_frame_info.tsv"


map_df = pd.read_csv(MAP_FPATH)
info_df = pd.read_csv(INFO_FPATH, sep = '\t', usecols = [0,1], header = None, skiprows= 1)
pb2va_dict = {pb:va for pb, va in zip(map_df["propbank"], map_df["verbatlas"])}
va2keys_dict = {va:key for va, key in zip(info_df[0], info_df[1])}


def pb_to_va(pb):
    if pb in pb2va_dict:
        return pb2va_dict[pb]
    return None


def va_to_key(va):
    if va in va2keys_dict:
        return va2keys_dict[va]
    return None