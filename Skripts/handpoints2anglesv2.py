import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
import sys
import yaml

def vectors2angle(v1, v2):
    # 3D vector inputs
    #TODO: Sine part leads to absolute values, therefore only angles in between 0 and 180° are returned, same result as with arccos but with a higher precission
    return np.arctan2(np.linalg.norm(np.cross(v1,v2,axis=0)), (v1.transpose()@v2)[0][0])

def handpoints2angles(handpoints):
    handpoints.reset_index(drop=True, inplace=True)
    # TODO kann optimiert werden indem normierte vektoren in vectors2angle eingaben vorrausgesetzt wird und diese direkt bei der v_andnormal und v_handbaseSpread berechnet wird. dann müssen aber auch die anderen vektoen direct normiert werden
    ts = handpoints.loc[0,'timestamp']
    coordinates = handpoints.loc[:,['point', 'x','y','z']].set_index('point')

    # calc normal vector of handpalm (vectors of shape (3,1))    
    v_handnormal = np.cross(np.array([coordinates.loc['Pinky_MCP'] - coordinates.loc['Wrist']]).transpose(), 
                            np.array([coordinates.loc['Index_MCP'] - coordinates.loc['Wrist']]).transpose(),axis=0)    # TODO umbenennen, bei jedem anderen datensatz passt es wieder nicht, da hand oftpalm oder wrist ist
    v_handbaseSpread = np.array([coordinates.loc['Index_MCP'] - coordinates.loc['Pinky_MCP']]).transpose()

    # calculate base coordinate system of the handpalm [v_handnormal, v_handbaseSpread, v_handnormal x v_handbaseSpread]
    mattrans = np.concatenate((v_handbaseSpread/np.linalg.norm(v_handbaseSpread), v_handnormal/np.linalg.norm(v_handnormal), np.cross(v_handbaseSpread, v_handnormal, axis=0)/np.linalg.norm(np.cross(v_handbaseSpread, v_handnormal, axis = 0))), axis = 1)
    q = Rotation.from_matrix(mattrans).as_quat()

    [thb_spread, thb_flex] = clc2dof_joint(mattrans, np.array([coordinates.loc['Thumb_MCP'] - coordinates.loc['Thumb_CMC']]).transpose())
    [idx_spread, idx_flex] = clc2dof_joint(mattrans, np.array([coordinates.loc['Index_PIP'] - coordinates.loc['Index_MCP']]).transpose())
    [mid_spread, mid_flex] = clc2dof_joint(mattrans, np.array([coordinates.loc['Middle_PIP'] - coordinates.loc['Middle_MCP']]).transpose())
    [ring_spread, ring_flex] = clc2dof_joint(mattrans, np.array([coordinates.loc['Ring_PIP'] - coordinates.loc['Ring_MCP']]).transpose())
    [pinky_spread, pinky_flex] = clc2dof_joint(mattrans, np.array([coordinates.loc['Pinky_PIP'] - coordinates.loc['Pinky_MCP']]).transpose())

    # [thb_spread, thb_flex] = clc2dof_joint_simple(mattrans, np.array([coordinates.loc['Thumb_MCP'] - coordinates.loc['Thumb_CMC']]).transpose())
    # [idx_spread, idx_flex] = clc2dof_joint_simple(mattrans, np.array([coordinates.loc['Index_PIP'] - coordinates.loc['Index_MCP']]).transpose())
    # [mid_spread, mid_flex] = clc2dof_joint_simple(mattrans, np.array([coordinates.loc['Middle_PIP'] - coordinates.loc['Middle_MCP']]).transpose())
    # [ring_spread, ring_flex] = clc2dof_joint_simple(mattrans, np.array([coordinates.loc['Ring_PIP'] - coordinates.loc['Ring_MCP']]).transpose())
    # [pinky_spread, pinky_flex] = clc2dof_joint_simple(mattrans, np.array([coordinates.loc['Pinky_PIP'] - coordinates.loc['Pinky_MCP']]).transpose())


    data = {
        # calculate angles of Thumb finger
        'timestamp': ts,

        'Thumb_CMC_Spread': thb_spread,
        'Thumb_CMC_Flex': thb_flex,
        'Thumb_MCP_Flex': vectors2angle(np.array([coordinates.loc['Thumb_MCP'] - coordinates.loc['Thumb_CMC']]).transpose(), 
                                         np.array([coordinates.loc['Thumb_IP'] - coordinates.loc['Thumb_MCP']]).transpose()),
        'Thumb_IP_Flex': vectors2angle(np.array([coordinates.loc['Thumb_IP'] - coordinates.loc['Thumb_MCP']]).transpose(), 
                                       np.array([coordinates.loc['Thumb_TIP'] - coordinates.loc['Thumb_IP']]).transpose()),

        # calculate angles of Index finger
        'Index_MCP_Spread': idx_spread,
        'Index_MCP_Flex': idx_flex,
        'Index_PIP_Flex': vectors2angle(np.array([coordinates.loc['Index_PIP'] - coordinates.loc['Index_MCP']]).transpose(), 
                                         np.array([coordinates.loc['Index_DIP'] - coordinates.loc['Index_PIP']]).transpose()),
        'Index_DIP_Flex': vectors2angle(np.array([coordinates.loc['Index_DIP'] - coordinates.loc['Index_PIP']]).transpose(), 
                                        np.array([coordinates.loc['Index_TIP'] - coordinates.loc['Index_DIP']]).transpose()),

        # calculate angles of Middle finger
        'Middle_MCP_Spread': mid_spread,
        'Middle_MCP_Flex': mid_flex,
        'Middle_PIP_Flex': vectors2angle(np.array([coordinates.loc['Middle_PIP'] - coordinates.loc['Middle_MCP']]).transpose(), 
                                          np.array([coordinates.loc['Middle_DIP'] - coordinates.loc['Middle_PIP']]).transpose()),
        'Middle_DIP_Flex': vectors2angle(np.array([coordinates.loc['Middle_DIP'] - coordinates.loc['Middle_PIP']]).transpose(), 
                                         np.array([coordinates.loc['Middle_TIP'] - coordinates.loc['Middle_DIP']]).transpose()),

        # calculate angles of Ring finger
        'Ring_MCP_Spread': ring_spread,
        'Ring_MCP_Flex': ring_flex,
        'Ring_PIP_Flex': vectors2angle(np.array([coordinates.loc['Ring_PIP'] - coordinates.loc['Ring_MCP']]).transpose(), 
                                        np.array([coordinates.loc['Ring_DIP'] - coordinates.loc['Ring_PIP']]).transpose()),
        'Ring_DIP_Flex': vectors2angle(np.array([coordinates.loc['Ring_DIP'] - coordinates.loc['Ring_PIP']]).transpose(), 
                                       np.array([coordinates.loc['Ring_TIP'] - coordinates.loc['Ring_DIP']]).transpose()),

        # calculate angles of Pinky finger
        'Pinky_MCP_Spread': pinky_spread,
        'Pinky_MCP_Flex': pinky_flex,
        'Pinky_PIP_Flex': vectors2angle(np.array([coordinates.loc['Pinky_PIP'] - coordinates.loc['Pinky_MCP']]).transpose(), 
                                        np.array([coordinates.loc['Pinky_DIP'] - coordinates.loc['Pinky_PIP']]).transpose()),
        'Pinky_DIP_Flex': vectors2angle(np.array([coordinates.loc['Pinky_DIP'] - coordinates.loc['Pinky_PIP']]).transpose(), 
                                       np.array([coordinates.loc['Pinky_TIP'] - coordinates.loc['Pinky_DIP']]).transpose()),


        #hand Frame
        'Handpoint_X': coordinates.loc['Wrist', 'x'],
        'Handpoint_Y': coordinates.loc['Wrist', 'y'],
        'Handpoint_Z': coordinates.loc['Wrist', 'z'],

        'Handpoint_Quaternion_v1': q[0],
        'Handpoint_Quaternion_v2': q[1],
        'Handpoint_Quaternion_v3': q[2],
        'Handpoint_Quaternion_s': q[3],

    }
    Dataframe = pd.DataFrame.from_records([data])
    Dataframe.loc[:,'Thumb_CMC_Spread':'Pinky_DIP_Flex'] = np.degrees(Dataframe.loc[:,'Thumb_CMC_Spread':'Pinky_DIP_Flex'])
    return Dataframe

def clc2dof_joint(mattrans, v_global):
        # calculation of the 2dof joint including a transformation of the vector between MCP and PIPfrom global to local coordinates
        v_trans = np.linalg.inv(mattrans) @ v_global

        spread = np.arctan2(v_trans[0][0], v_trans[2][0])
        flex = np.arccos(v_trans[1][0]/np.linalg.norm(v_trans)) - np.pi/2
        if spread > np.pi/2:
            spread = spread - np.pi
            flex = np.pi - flex
        elif spread < -np.pi/2:
            spread = spread + np.pi
            flex = np.pi - flex
        
        return [spread, flex]

def clc2dof_joint_simple(mattrans, v_global):
    # calculation of the 2dof joint including a transformation of the vector between MCP and PIPfrom global to local coordinates
    v_trans = np.linalg.inv(mattrans) @ v_global

    spread = np.arctan2(v_trans[0][0], v_trans[2][0])
    flex = np.arccos(v_trans[1][0]/np.linalg.norm(v_trans)) - np.pi/2
    
    return [spread, flex]
    
def pt2vec(handpoints, pt_start, pt_end):
    v_t = (np.array(handpoints[handpoints['point'] == pt_end].loc[:, ['x', 'y', 'z']]) - np.array(handpoints[handpoints['point'] == pt_start].loc[:, ['x', 'y', 'z']]))
    return v_t.transpose()

# def pt2vecnorm(handpoints, pt_start, pt_end):
#     pt1 = handpoints[handpoints['point'] == pt_end]
#     pt2 = handpoints[handpoints['point'] == pt_start]
#     v_t = (np.array(pt1.loc[:, ['x', 'y', 'z']]) - np.array(pt2.loc[:, ['x', 'y', 'z']]))
#     v_t = v_t / np.linalg.norm(v_t)
#     return v_t.transpose()


def pt2vecnorm(pt_start, pt_end):
    v_t = (np.array([pt_end - pt_start]))
    v_t = v_t / np.linalg.norm(v_t)
    return v_t.transpose()

# pt2vecnorm(handpoints, 'Wrist', 'Thumb_CMC')
# pt2vecnorm(coordinates.loc['Wrist'], coordinates.loc['Thumb_CMC'])

def handpoints2vectors(handpoints):
    handpoints.reset_index(drop=True, inplace=True)
    # TODO kann optimiert werden indem normierte vektoren in vectors2angle eingaben vorrausgesetzt wird und diese direkt bei der v_andnormal und v_handbaseSpread berechnet wird. dann müssen aber auch die anderen vektoen direct normiert werden
    ts = handpoints.loc[0,'timestamp']
    coordinates = handpoints.loc[:,['point', 'x','y','z']].set_index('point')
        
    v_handnormal = np.cross(np.array([coordinates.loc['Pinky_MCP'] - coordinates.loc['Wrist']]).transpose(), 
                            np.array([coordinates.loc['Index_MCP'] - coordinates.loc['Wrist']]).transpose(),axis=0)    # TODO umbenennen, bei jedem anderen datensatz passt es wieder nicht, da hand oftpalm oder wrist ist
    v_handbaseSpread = np.array([coordinates.loc['Index_MCP'] - coordinates.loc['Pinky_MCP']]).transpose()

    mattrans = np.concatenate((v_handbaseSpread/np.linalg.norm(v_handbaseSpread), v_handnormal/np.linalg.norm(v_handnormal), np.cross(v_handbaseSpread, v_handnormal, axis=0)/np.linalg.norm(np.cross(v_handbaseSpread, v_handnormal, axis = 0))), axis = 1)
    q = Rotation.from_matrix(mattrans).as_quat()
    ####

    thumb_cmc = pt2vecnorm(coordinates.loc['Thumb_CMC'], coordinates.loc['Thumb_MCP']) - pt2vecnorm(coordinates.loc['Wrist'], coordinates.loc['Thumb_CMC'])
    thumb_cmc = thumb_cmc / np.linalg.norm(thumb_cmc)
    thumb_mcp = pt2vecnorm(coordinates.loc['Thumb_MCP'], coordinates.loc['Thumb_DIP']) - pt2vecnorm(coordinates.loc['Thumb_CMC'], coordinates.loc['Thumb_MCP'])
    thumb_mcp = thumb_mcp / np.linalg.norm(thumb_mcp)
    thumb_dip = pt2vecnorm(coordinates.loc['Thumb_DIP'], coordinates.loc['Thumb_TIP']) - pt2vecnorm(coordinates.loc['Thumb_MCP'], coordinates.loc['Thumb_DIP'])
    thumb_dip = thumb_dip / np.linalg.norm(thumb_dip)

    index_mcp = pt2vecnorm(coordinates.loc['Index_MCP'], coordinates.loc['Index_PIP']) - pt2vecnorm(coordinates.loc['Wrist'], coordinates.loc['Index_MCP'])
    index_mcp = index_mcp / np.linalg.norm(index_mcp)
    index_pip = pt2vecnorm(coordinates.loc['Index_PIP'], coordinates.loc['Index_DIP']) - pt2vecnorm(coordinates.loc['Index_MCP'], coordinates.loc['Index_PIP'])
    index_pip = index_pip / np.linalg.norm(index_pip)
    index_dip = pt2vecnorm(coordinates.loc['Index_DIP'], coordinates.loc['Index_TIP']) - pt2vecnorm(coordinates.loc['Index_PIP'], coordinates.loc['Index_DIP'])
    index_dip = index_dip / np.linalg.norm(index_dip)

    middle_mcp = pt2vecnorm(coordinates.loc['Middle_MCP'], coordinates.loc['Middle_PIP']) - pt2vecnorm(coordinates.loc['Wrist'], coordinates.loc['Middle_MCP'])
    middle_mcp = middle_mcp / np.linalg.norm(middle_mcp)
    middle_pip = pt2vecnorm(coordinates.loc['Middle_PIP'], coordinates.loc['Middle_DIP']) - pt2vecnorm(coordinates.loc['Middle_MCP'], coordinates.loc['Middle_PIP'])
    middle_pip = middle_pip / np.linalg.norm(middle_pip)
    middle_dip = pt2vecnorm(coordinates.loc['Middle_DIP'], coordinates.loc['Middle_TIP']) - pt2vecnorm(coordinates.loc['Middle_PIP'], coordinates.loc['Middle_DIP'])
    middle_dip = middle_dip / np.linalg.norm(middle_dip)

    ring_mcp = pt2vecnorm(coordinates.loc['Ring_MCP'], coordinates.loc['Ring_PIP']) - pt2vecnorm(coordinates.loc['Wrist'], coordinates.loc['Ring_MCP'])
    ring_mcp = ring_mcp / np.linalg.norm(ring_mcp)
    ring_pip = pt2vecnorm(coordinates.loc['Ring_PIP'], coordinates.loc['Ring_DIP']) - pt2vecnorm(coordinates.loc['Ring_MCP'], coordinates.loc['Ring_PIP'])
    ring_pip = ring_pip / np.linalg.norm(ring_pip)
    ring_dip = pt2vecnorm(coordinates.loc['Ring_DIP'], coordinates.loc['Ring_TIP']) - pt2vecnorm(coordinates.loc['Ring_PIP'], coordinates.loc['Ring_DIP'])
    ring_dip = ring_dip / np.linalg.norm(ring_dip)

    pinky_mcp = pt2vecnorm(coordinates.loc['Pinky_MCP'], coordinates.loc['Pinky_PIP']) - pt2vecnorm(coordinates.loc['Wrist'], coordinates.loc['Pinky_MCP'])
    pinky_mcp = pinky_mcp / np.linalg.norm(pinky_mcp)
    pinky_pip = pt2vecnorm(coordinates.loc['Pinky_PIP'], coordinates.loc['Pinky_DIP']) - pt2vecnorm(coordinates.loc['Pinky_MCP'], coordinates.loc['Pinky_PIP'])
    pinky_pip = pinky_pip / np.linalg.norm(pinky_pip)
    pinky_dip = pt2vecnorm(coordinates.loc['Pinky_DIP'], coordinates.loc['Pinky_TIP']) - pt2vecnorm(coordinates.loc['Pinky_PIP'], coordinates.loc['Pinky_DIP'])
    pinky_dip = pinky_dip / np.linalg.norm(pinky_dip)

    data = {
        'timestamp': ts,
        'Thumb_CMC_X': thumb_cmc[0,0],
        'Thumb_CMC_Y': thumb_cmc[1,0],
        'Thumb_CMC_Z': thumb_cmc[2,0],
        'Thumb_MCP_X': thumb_mcp[0,0],
        'Thumb_MCP_Y': thumb_mcp[1,0],
        'Thumb_MCP_Z': thumb_mcp[2,0],
        'Thumb_DIP_X': thumb_dip[0,0],
        'Thumb_DIP_Y': thumb_dip[1,0],
        'Thumb_DIP_Z': thumb_dip[2,0],

        'Index_MCP_X': index_mcp[0,0],
        'Index_MCP_Y': index_mcp[1,0],
        'Index_MCP_Z': index_mcp[2,0],
        'Index_PIP_X': index_pip[0,0],
        'Index_PIP_Y': index_pip[1,0],
        'Index_PIP_Z': index_pip[2,0],
        'Index_DIP_X': index_dip[0,0],
        'Index_DIP_Y': index_dip[1,0],
        'Index_DIP_Z': index_dip[2,0],

        'Middle_MCP_X': middle_mcp[0,0],
        'Middle_MCP_Y': middle_mcp[1,0],
        'Middle_MCP_Z': middle_mcp[2,0],
        'Middle_PIP_X': middle_pip[0,0],
        'Middle_PIP_Y': middle_pip[1,0],
        'Middle_PIP_Z': middle_pip[2,0],
        'Middle_DIP_X': middle_dip[0,0],
        'Middle_DIP_Y': middle_dip[1,0],
        'Middle_DIP_Z': middle_dip[2,0],

        'Ring_MCP_X': ring_mcp[0,0],
        'Ring_MCP_Y': ring_mcp[1,0],
        'Ring_MCP_Z': ring_mcp[2,0],
        'Ring_PIP_X': ring_pip[0,0],
        'Ring_PIP_Y': ring_pip[1,0],
        'Ring_PIP_Z': ring_pip[2,0],
        'Ring_DIP_X': ring_dip[0,0],
        'Ring_DIP_Y': ring_dip[1,0],
        'Ring_DIP_Z': ring_dip[2,0],

        'Pinky_MCP_X': pinky_mcp[0,0],
        'Pinky_MCP_Y': pinky_mcp[1,0],
        'Pinky_MCP_Z': pinky_mcp[2,0],
        'Pinky_PIP_X': pinky_pip[0,0],
        'Pinky_PIP_Y': pinky_pip[1,0],
        'Pinky_PIP_Z': pinky_pip[2,0],
        'Pinky_DIP_X': pinky_dip[0,0],
        'Pinky_DIP_Y': pinky_dip[1,0],
        'Pinky_DIP_Z': pinky_dip[2,0],

        #hand Frame
        'Handpoint_X': coordinates.loc['Wrist', 'x'],
        'Handpoint_Y': coordinates.loc['Wrist', 'y'],
        'Handpoint_Z': coordinates.loc['Wrist', 'z'],

        'Handpoint_Quaternion_a': q[0],
        'Handpoint_Quaternion_b': q[1],
        'Handpoint_Quaternion_c': q[2],
        'Handpoint_Quaternion_d': q[3],

    }, 

    return data

def handpoints2vectors2(handpoints):
    handpoints.reset_index(drop=True, inplace=True)
    # TODO kann optimiert werden indem normierte vektoren in vectors2angle eingaben vorrausgesetzt wird und diese direkt bei der v_andnormal und v_handbaseSpread berechnet wird. dann müssen aber auch die anderen vektoen direct normiert werden
    ts = handpoints.loc[0,'timestamp']
    coordinates = handpoints.loc[:,['point', 'x','y','z']].set_index('point')
        
    v_handnormal = np.cross(np.array([coordinates.loc['Pinky_MCP'] - coordinates.loc['Wrist']]).transpose(), 
                            np.array([coordinates.loc['Index_MCP'] - coordinates.loc['Wrist']]).transpose(),axis=0)    # TODO umbenennen, bei jedem anderen datensatz passt es wieder nicht, da hand oftpalm oder wrist ist
    v_handbaseSpread = np.array([coordinates.loc['Index_MCP'] - coordinates.loc['Pinky_MCP']]).transpose()

    mattrans = np.concatenate((v_handbaseSpread/np.linalg.norm(v_handbaseSpread), v_handnormal/np.linalg.norm(v_handnormal), np.cross(v_handbaseSpread, v_handnormal, axis=0)/np.linalg.norm(np.cross(v_handbaseSpread, v_handnormal, axis = 0))), axis = 1)
    q = Rotation.from_matrix(mattrans).as_quat()
    ####

    thumb_cmc = np.linalg.inv(mattrans) @ pt2vecnorm(coordinates.loc['Thumb_CMC'], coordinates.loc['Thumb_MCP'])
    thumb_mcp = np.linalg.inv(mattrans) @ pt2vecnorm(coordinates.loc['Thumb_MCP'], coordinates.loc['Thumb_DIP'])
    thumb_dip = np.linalg.inv(mattrans) @ pt2vecnorm(coordinates.loc['Thumb_DIP'], coordinates.loc['Thumb_TIP'])

    index_mcp = np.linalg.inv(mattrans) @ pt2vecnorm(coordinates.loc['Index_MCP'], coordinates.loc['Index_PIP'])
    index_pip = np.linalg.inv(mattrans) @ pt2vecnorm(coordinates.loc['Index_PIP'], coordinates.loc['Index_DIP'])
    index_dip = np.linalg.inv(mattrans) @ pt2vecnorm(coordinates.loc['Index_DIP'], coordinates.loc['Index_TIP'])

    middle_mcp = np.linalg.inv(mattrans) @ pt2vecnorm(coordinates.loc['Middle_MCP'], coordinates.loc['Middle_PIP'])
    middle_pip = np.linalg.inv(mattrans) @ pt2vecnorm(coordinates.loc['Middle_PIP'], coordinates.loc['Middle_DIP'])
    middle_dip = np.linalg.inv(mattrans) @ pt2vecnorm(coordinates.loc['Middle_DIP'], coordinates.loc['Middle_TIP'])

    ring_mcp = np.linalg.inv(mattrans) @ pt2vecnorm(coordinates.loc['Ring_MCP'], coordinates.loc['Ring_PIP'])
    ring_pip = np.linalg.inv(mattrans) @ pt2vecnorm(coordinates.loc['Ring_PIP'], coordinates.loc['Ring_DIP'])
    ring_dip = np.linalg.inv(mattrans) @ pt2vecnorm(coordinates.loc['Ring_DIP'], coordinates.loc['Ring_TIP'])

    pinky_mcp = np.linalg.inv(mattrans) @ pt2vecnorm(coordinates.loc['Pinky_MCP'], coordinates.loc['Pinky_PIP'])
    pinky_pip = np.linalg.inv(mattrans) @ pt2vecnorm(coordinates.loc['Pinky_PIP'], coordinates.loc['Pinky_DIP'])
    pinky_dip = np.linalg.inv(mattrans) @ pt2vecnorm(coordinates.loc['Pinky_DIP'], coordinates.loc['Pinky_TIP'])

    data = {
        'timestamp': ts,
        'Thumb_CMC_X': thumb_cmc[0,0],
        'Thumb_CMC_Y': thumb_cmc[1,0],
        'Thumb_CMC_Z': thumb_cmc[2,0],
        'Thumb_MCP_X': thumb_mcp[0,0],
        'Thumb_MCP_Y': thumb_mcp[1,0],
        'Thumb_MCP_Z': thumb_mcp[2,0],
        'Thumb_DIP_X': thumb_dip[0,0],
        'Thumb_DIP_Y': thumb_dip[1,0],
        'Thumb_DIP_Z': thumb_dip[2,0],

        'Index_MCP_X': index_mcp[0,0],
        'Index_MCP_Y': index_mcp[1,0],
        'Index_MCP_Z': index_mcp[2,0],
        'Index_PIP_X': index_pip[0,0],
        'Index_PIP_Y': index_pip[1,0],
        'Index_PIP_Z': index_pip[2,0],
        'Index_DIP_X': index_dip[0,0],
        'Index_DIP_Y': index_dip[1,0],
        'Index_DIP_Z': index_dip[2,0],

        'Middle_MCP_X': middle_mcp[0,0],
        'Middle_MCP_Y': middle_mcp[1,0],
        'Middle_MCP_Z': middle_mcp[2,0],
        'Middle_PIP_X': middle_pip[0,0],
        'Middle_PIP_Y': middle_pip[1,0],
        'Middle_PIP_Z': middle_pip[2,0],
        'Middle_DIP_X': middle_dip[0,0],
        'Middle_DIP_Y': middle_dip[1,0],
        'Middle_DIP_Z': middle_dip[2,0],

        'Ring_MCP_X': ring_mcp[0,0],
        'Ring_MCP_Y': ring_mcp[1,0],
        'Ring_MCP_Z': ring_mcp[2,0],
        'Ring_PIP_X': ring_pip[0,0],
        'Ring_PIP_Y': ring_pip[1,0],
        'Ring_PIP_Z': ring_pip[2,0],
        'Ring_DIP_X': ring_dip[0,0],
        'Ring_DIP_Y': ring_dip[1,0],
        'Ring_DIP_Z': ring_dip[2,0],

        'Pinky_MCP_X': pinky_mcp[0,0],
        'Pinky_MCP_Y': pinky_mcp[1,0],
        'Pinky_MCP_Z': pinky_mcp[2,0],
        'Pinky_PIP_X': pinky_pip[0,0],
        'Pinky_PIP_Y': pinky_pip[1,0],
        'Pinky_PIP_Z': pinky_pip[2,0],
        'Pinky_DIP_X': pinky_dip[0,0],
        'Pinky_DIP_Y': pinky_dip[1,0],
        'Pinky_DIP_Z': pinky_dip[2,0],

        #hand Frame
        'Handpoint_X': coordinates.loc['Wrist', 'x'],
        'Handpoint_Y': coordinates.loc['Wrist', 'y'],
        'Handpoint_Z': coordinates.loc['Wrist', 'z'],

        'Handpoint_Quaternion_a': q[0],
        'Handpoint_Quaternion_b': q[1],
        'Handpoint_Quaternion_c': q[2],
        'Handpoint_Quaternion_d': q[3],

    }, 

    return data

if __name__ == "__main__":

    DHG3D_test = pd.DataFrame({
        'timestamp': np.zeros(22), 
        'point': [ "Wrist", "Hand", "Thumb_CMC", "Thumb_MCP", "Thumb_DIP", "Thumb_TIP", "Index_MCP", "Index_PIP", "Index_DIP", "Index_TIP", "Middle_MCP", "Middle_PIP", "Middle_DIP", "Middle_TIP", "Ring_MCP", "Ring_PIP", "Ring_DIP", "Ring_TIP", "Pinky_MCP", "Pinky_PIP", "Pinky_DIP", "Pinky_TIP"],
        'x':     [ 0, 0, -1,-1,-1,-1,-2,-2,-2,-2, -1, -1, -1, -1, 1, 0.5, 0.25, 0, 2, 3, 3, 3 ],
        'y':     [ 0, 1,0.5, 1, 1, 1, 3, 5, 5, 4,  3,  5,  6,  7, 3,  5,   6,   7, 3,  5,  5, 4],
        'z':     [ 0, 0,  0, 0, 0, 0, 0,-2,-3,-4,  0,  0,  0,  0, 0,  0,   0,   0,  0, np.sqrt(5),  0,  0]
        })

    handpoints = DHG3D_test[DHG3D_test['timestamp'] == 0.0]
    data = handpoints2angles(handpoints)
    data['angle'] = np.degrees(data['angle'])

    print(data)