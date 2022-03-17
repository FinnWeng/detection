import numpy as np

def check_intersection(intv_a, intv_b):
    a_max_greaterthan_b_min = a[:,1] > b[:,0]
    b_max_greaterthan_a_min = b[:,1] > a[:,0]
    a_max_greaterthan_b_max = a[:,1] > b[:,1]
    a_min_greaterthan_b_min = a[:,0] > b[:,0]

    '''
    decide whether they overlap
    '''
    overlap = np.logical_and(a_max_greaterthan_b_min, b_max_greaterthan_a_min)
    print(overlap)

    '''
    decide how to compute
    '''
    case_1 = np.logical_and( np.invert(a_max_greaterthan_b_max), np.invert(a_min_greaterthan_b_min)).astype(int)
    case_2 = np.logical_and( a_max_greaterthan_b_max, a_min_greaterthan_b_min).astype(int)
    case_3 = np.logical_and( np.invert(a_max_greaterthan_b_max), a_min_greaterthan_b_min).astype(int)
    case_4 = np.logical_and( a_max_greaterthan_b_max, np.invert(a_min_greaterthan_b_min)).astype(int)

    print("case_1:", case_1)
    print("case_2:", case_2)
    print("case_3:", case_3)
    print("case_4:", case_4)
    
    case_1_area = (a[:,1] - b[:,0]) * case_1
    case_2_area = (b[:,1] - a[:,0]) * case_2
    case_3_area = (a[:,1] - a[:,0]) * case_3
    case_4_area = (b[:,1] - b[:,0]) * case_4

    

    overlap_area = case_1_area + case_2_area + case_3_area + case_4_area
    overlap_area = overlap.astype(int)*overlap_area


    return overlap_area
    


if __name__ == "__main__":
    a = np.array([[10,15],[10,15]])
    b = np.array([[10,15],[10,15]])
    print(check_intersection(a,b))