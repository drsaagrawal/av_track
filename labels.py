#!/usr/bin/python
#
#@Author https://github.com/Amnatehreem

from collections import namedtuple

Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'color'       , # The color of this label
    ] )

labels = [
    #       name                                  color
    Label(  'unlabeled'                         , (  0,  0,  0) ),
    Label(  'ego '                              , (  0,  0,  0) ),
    Label(  'rectification border'              , (  0,  0,  0) ),
    Label(  'out of roi'                        , (  0,  0,  0) ),
    Label(  'static'                            , (  0,  0,  0) ),
    Label(  'dynamic'                           , (111, 74,  0) ),
    Label(  'ground'                            , ( 81,  0, 81) ),
    Label(  'road'                              , (128, 64,128) ),
    Label(  'sidewalk'                          , (244, 35,232) ),
    Label(  'parking'                           , (250,170,160) ),
    Label(  'rail track'                        , (230,150,140) ),
    Label(  'building'                          , ( 70, 70, 70) ),
    Label(  'wall'                              , (102,102,156) ),
    Label(  'fence'                             , (190,153,153) ),
    Label(  'guard rail'                        , (180,165,180) ),
    Label(  'bridge'                            , (150,100,100) ),
    Label(  'tunnel'                            , (150,120, 90) ),
    Label(  'pole'                              , (153,153,153) ),
    Label(  'polegroup'                         , (153,153,153) ),
    Label(  'traffic light'                     , (250,170, 30) ),
    Label(  'traffic sign'                      , (220,220,  0) ),
    Label(  'vegetation'                        , (107,142, 35) ),
    Label(  'terrain'                           , (152,251,152) ),
    Label(  'sky'                               , (119, 69,100) ),
    Label(  'person'                            , (150, 20, 60) ),
    Label(  'rider'                             , (255,  0,  0) ),
    Label(  'car'                               , (  0,  0,142) ),
    Label(  'truck'                             , (  0,  0, 70) ),
    Label(  'bus'                               , (  0, 60,100) ),
    Label(  'caravan'                           , (  0,  0, 90) ),
    Label(  'trailer'                           , (250,170, 30) ),
    Label(  'train'                             , ( 20, 80,100) ),
    Label(  'motorcycle'                        , (  0,  0,230) ),
    Label(  'bicycle'                           , (101,114,135) ),
    Label(  'license plate'                     , (68, 104,163) ),
    Label(  'unlabeled'                         , (245,137,255) ),
    Label(  'ego '                              , (192,255,  0) ),
    Label(  'rectification border'              , (214,  4,168) ),
    Label(  'out of roi'                        , (104,  2,  2) ),
    Label(  'static'                            , ( 65,  0,119) ),
    Label(  'dynamic'                           , ( 85,104, 20) ),
    Label(  'ground'                            , ( 80, 80,170) ),
    Label(  'road'                              , (255,  0, 80) ),
    Label(  'sidewalk'                          , (229,152,119) ),
    Label(  'parking'                           , (203, 61,255) ),
    Label(  'rail track'                        , (175,111, 59) ),
    Label(  'building'                          , ( 23,237, 11) ),
    Label(  'wall'                              , ( 74, 76,124) ),
    Label(  'fence'                             , (219,210,213) ),
    Label(  'guard rail'                        , (249, 67,161) ),
    Label(  'bridge'                            , (225,247,153) ),
    Label(  'tunnel'                            , (255,202, 86) ),
    Label(  'pole'                              , (214,165,255) ),
    Label(  'polegroup'                         , ( 39,102, 59) ),
    Label(  'traffic light'                     , (126, 60,132) ),
    Label(  'traffic sign'                      , (244,196,176) ),
    Label(  'vegetation'                        , (247,136,223) ),
    Label(  'terrain'                           , (163,247,215) ),
    Label(  'sky'                               , (214,  4,168) ),
    Label(  'person'                            , (153,247,154) ),
    Label(  'rider'                             , (244,161,204) ),
    Label(  'car'                               , (144,249,244) ),
    Label(  'truck'                             , (  0,  0,183) ),
    Label(  'bus'                               , (249, 74,  0) ),
    Label(  'caravan'                           , (168, 26, 99) ),
    Label(  'trailer'                           , ( 73,153,149) ),
    Label(  'train'                             , ( 45, 41, 43) ),
    Label(  'motorcycle'                        , (220, 99,121) ),
    Label(  'bicycle'                           , (232,191,111) ),
    Label(  'license plate'                     , (140, 65,104) ),
    Label(  'unlabeled'                         , (  5,200,255) ),
    Label(  'ego '                              , (119,  4,160) ),
    Label(  'ego '                              , (255,  2,196) ),
]

