import os
import datetime
import sys, getopt
def log_current_time(tag_str, status):
    timefmt = "%Y-%m%d-%H:%M:%S.%f"
    now = datetime.datetime.now()
    nowtime_str = now.strftime(timefmt)
    print("[{}/{}] = {}".format( tag_str, status,  nowtime_str ))
def generate_rs_by_ct_folder(input_ct_folder, output_rs_filepath, model_name, is_recreate_ai_bytes = True, bytes_filepath="mrcnn_output.bytes"):
    """
    Generate RS file by the folder that include CT files
    :param input_ct_folder:
    :param output_rs_filepath:
    :param model_name:
    :return:
    """
    from SimpleInterpolateRsWrapUp import interpolate_and_wrapup_rs
    from AI_process import AI_process_get_predict_result
    from utilities import python_object_dump
    from utilities import python_object_load
    import pydicom
    ct_folder = input_ct_folder
    ct_filelist = []
    for file in os.listdir(ct_folder):
        filepath = os.path.join(ct_folder, file)
        try:
            ct_fp = pydicom.read_file(filepath)
            if ct_fp.Modality == 'CT':
                ct_filelist.append(filepath)
        except:
            pass
    print("len of ct_filelist = ", len(ct_filelist))
    log_current_time("AI_process", "START")
    # mrcnn_out = ai.AI_process_by_folder(ct_folder, model_name)
    #mrcnn_out = ai.AI_process(ct_filelist, model_name)
    mrcnn_out = None
    bytes_file_exists = os.path.exists(bytes_filepath)
    if bytes_file_exists == False:
        mrcnn_out = AI_process_get_predict_result(ct_filelist, model_name)
        python_object_dump(mrcnn_out, bytes_filepath)
    else: # case bytes_file_exists == True
        if is_recreate_ai_bytes == True:
            mrcnn_out = AI_process_get_predict_result(ct_filelist, model_name)
            python_object_dump(mrcnn_out, bytes_filepath)
        else: # Case is_create_ai_byte == False and bytes_file_exists == True
            mrcnn_out = python_object_load(bytes_filepath)

        #mrcnn_out = AI_process_get_predict_result(ct_filelist, model_name)

    #print('EARLY exit(1) in generate_rs_by_ct_folder()')
    #exit(1)

    log_current_time("AI_process", "STOP")
    #print(mrcnn_out)

    log_current_time("InterpolateWrapper_process", "START")
    #sirw.interpolate_and_wrapup_rs(mrcnn_out, ct_filelist, "RS.output.dcm")
    interpolate_and_wrapup_rs(mrcnn_out, ct_filelist, output_rs_filepath, model_name)
    #os.path.join(temp_folder, r'RS.output.dcm')
    log_current_time("InterpolateWrapper_process", "STOP")
def generate_rp_by_ct_rs_folder(input_ct_rs_folder, output_rp_filepath):
    """
    Generate output RP file by the folder that include CT files and RS file
    :param input_ct_rs_folder:
    :param output_rp_filepath:
    :return:
    """
    from utilities import generate_metadata_to_dicom_dict
    from generate_rp_brachy_in_batch import get_dicom_dict
    from generate_rp_brachy_in_batch import generate_brachy_rp_file
    from generate_rp_brachy_in_batch import generate_output_to_dicom_dict
    folder = input_ct_rs_folder
    dicom_dict = get_dicom_dict(folder)
    generate_metadata_to_dicom_dict(dicom_dict)
    generate_output_to_dicom_dict(dicom_dict)
    generate_brachy_rp_file(RP_OperatorsName='cylin',
                            dicom_dict=dicom_dict,
                            out_rp_filepath=output_rp_filepath,
                            is_enable_print=False)
def generate_rs_rp_by_ct_folder(input_ct_folder, output_rs_rp_folder, model_name):
    def clean_all_files_in_folder(folder):
        import shutil
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            try:
                if os.path.isfile(filepath) or os.path.islink(filepath):
                    os.unlink(filepath)
                elif os.path.isdir(filepath):
                    shutil.rmtree(filepath)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (filepath, e))
    from utilities import create_directory_if_not_exists
    from shutil import copyfile
    import pydicom
    model_name = "MRCNN_Brachy"
    input_folder = input_ct_folder
    output_folder = output_rs_rp_folder
    temp_folder = r"temp"
    create_directory_if_not_exists(temp_folder)
    clean_all_files_in_folder(temp_folder)
    # Step 1. copy all ct file from input folder to temp folder
    for file in os.listdir(input_folder):
        filepath = os.path.join(input_folder, file)
        try:
            fp = pydicom.read_file(filepath)
            if fp.Modality == 'CT':
                src_ct_filepath = filepath
                dst_ct_filepath = os.path.join(temp_folder, os.path.basename(filepath))
                copyfile(src_ct_filepath, dst_ct_filepath)
        except:
            pass

    temp_rs_filepath = os.path.join(temp_folder, r'RS.output.dcm')
    temp_rp_filepath = os.path.join(temp_folder, r'RP.output.dcm')
    #Step 2. gen rs in the temp folder
    generate_rs_by_ct_folder(
        input_ct_folder=temp_folder,
        output_rs_filepath=temp_rs_filepath,
        model_name="MRCNN_Brachy"
    )

    # Step 3. gen rp in the temp folder
    generate_rp_by_ct_rs_folder(
        input_ct_rs_folder=temp_folder,
        output_rp_filepath=temp_rp_filepath
    )

    # Step 4. copy rs and rp file into output_folder
    rs_src_filepath = temp_rs_filepath
    rs_dst_filepath = os.path.join(output_folder, os.path.basename(temp_rs_filepath))
    copyfile(rs_src_filepath, rs_dst_filepath)
    rp_src_filepath = temp_rp_filepath
    rp_dst_filepath = os.path.join(output_folder, os.path.basename(temp_rp_filepath))
    copyfile(rp_src_filepath, rp_dst_filepath)
    print("rs_dst_filepath = {}".format(rs_dst_filepath))
    print("rp_dst_filepath = {}".format(rp_dst_filepath))


def generate_rs_rd_by_ct_folder(input_ct_folder, output_rs_rd_folder, model_name):
    #TODO not done yet
    model_name = "MRCNN_Breast"
    #input_folder = "TestCase_Breast_Input_CtFolder"
    input_folder = input_ct_folder
    generate_rs_by_ct_folder(
        input_ct_folder=input_folder,
        output_rs_filepath=os.path.join(input_folder, r'RS.output.dcm'),
        model_name=model_name,
        is_recreate_ai_bytes=True
    )


def dev_test_code_running():

    def example_of_export_from_patient():
        from shutil import copyfile
        from utilities import create_directory_if_not_exists
        import pydicom
        import os
        from generate_rd_file import generate_rd_by_ct_rs
        patient_number = r"26896072"

        root_folder = r"BatchShowCaseFolder"
        output_root_folder = r"BatchOutput"
        patient_input_folder = os.path.join(root_folder, patient_number)
        patient_output_folder = os.path.join(output_root_folder, patient_number)
        create_directory_if_not_exists(patient_output_folder)
        # Make dcm filelist
        dcm_filelist = []
        for dirpath, subdirs, files in os.walk(patient_input_folder):
            for x in files:
                if x.endswith(".dcm"):
                    dcm_filelist.append(os.path.join(dirpath, x))
        # Find ct filelist and rs filepath
        ct_filelist = []
        rs_filepath = None
        for filepath in dcm_filelist:
            fp = pydicom.read_file(filepath)
            if fp.Modality == "CT":
                ct_filelist.append(filepath)
                basename = os.path.basename(filepath)
                copyfile(filepath, os.path.join(patient_output_folder, basename))
            elif fp.Modality == "RTSTRUCT":
                rs_filepath = filepath
                basename = os.path.basename(filepath)
                copyfile(filepath, os.path.join(patient_output_folder, basename))
        # predict to RS file by CT
        model_name = "MRCNN_Breast"
        input_folder = "TestCase_Breast_Input_CtFolder"
        generate_rs_by_ct_folder(
            input_ct_folder=patient_output_folder,
            output_rs_filepath=os.path.join(patient_output_folder, r'RS.output.dcm'),
            model_name=model_name,
            is_recreate_ai_bytes = True
        )
        output_rd_filepath = os.path.join(patient_output_folder, "rd.output.dcm")
        bytes_filepath = os.path.join(patient_output_folder, "rd.output.bytes")

        # predict to RD file by RS and CT
        generate_rd_by_ct_rs(rs_filepath, ct_filelist, output_rd_filepath, is_recreate=True, bytes_filepath=bytes_filepath)
        
    example_of_export_from_patient()


    # example code of how to gen RS from CT folder
    def example_of_gen_breast_rs():
        model_name = "MRCNN_Breast"
        input_folder = "TestCase_Breast_Input_CtFolder"
        generate_rs_by_ct_folder(
            input_ct_folder=input_folder,
            output_rs_filepath=os.path.join(input_folder, r'RS.output.dcm'),
            model_name=model_name,
            is_recreate_ai_bytes = True
        )
    #example_of_gen_breast_rs()

    def example_of_gen_breast_rd():
        from generate_rd_file import generate_rd_by_ct_rs
        import pydicom
        rs_filepath = r"./TestCase_Breast_Input_CtFolder/RS.output.dcm"
        #ct_folder_filepath = r"./TestCase_Breast_Input_CtFolder"

        #rs_filepath = r"Frankie_Dataset\24120779\C1_20181031\C1Bre4256\Structure\RS.1.2.246.352.71.4.417454940236.250260.20190418131658.dcm"
        #ct_folder_filepath = r"Frankie_Dataset\24120779\C1_20181031\C1Bre4256\CT"

        ct_folder_filepath = r"./TestCase_Breast_Input_CtFolder/"
        #ct_folder_filepath = r"PerfectTraining\34171876\C1_20181108\C1Bre4256\CT"
        #rs_filepath = r"PerfectTraining\34171876\C1_20181108\C1Bre4256\Structure\RS.1.2.246.352.71.4.417454940236.250903.20190418140351.dcm"

        bytes_filepath = r"new-mask100.bytes"
        output_rd_filepath = r"./TestCase_Breast_Input_CtFolder/rd.ai.output.dcm"
        ct_filelist = []
        for file in os.listdir(ct_folder_filepath):
            filepath = os.path.join(ct_folder_filepath, file)
            try:
                fp = pydicom.read_file(filepath)
                if fp.Modality == 'CT':
                    ct_filelist.append(filepath)
            except:
                continue
        generate_rd_by_ct_rs(rs_filepath, ct_filelist, output_rd_filepath, is_recreate=True, bytes_filepath=bytes_filepath)
    #example_of_gen_breast_rd()



    def example_of_gen_brachy_rs():
        model_name = "MRCNN_Brachy"
        input_folder = "TestCase_Brachy_Input_CtFolder"
        generate_rs_by_ct_folder(
            input_ct_folder=input_folder,
            output_rs_filepath=os.path.join(input_folder, r'RS.output.dcm'),
            model_name=model_name)
    #example_of_gen_brachy_rs()

    # example code of how to gen RP from CT RS folder
    def example_of_gen_rp():
        generate_rp_by_ct_rs_folder(
            input_ct_rs_folder=r"RAL_plan_new_20190905\29059811-1",
            output_rp_filepath=r"RP.output.dcm")
    # example code of how to gen RS & RP from CT folder
    def example_of_gen_rs_rp():
        input_folder = r"ShowCase01Test-Input-29059811"
        output_folder = r"ShowCase01Test-Output-29059811"
        generate_rs_rp_by_ct_folder(
            input_ct_folder=input_folder,
            output_rs_rp_folder=output_folder,
            model_name="MRCNN_Brachy")
    #example_of_gen_rs_rp()


    pass

def main(argv):
    inputfolder = ""
    outputfolder = ""
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["inputfolder=","outputfolder="])
    except getopt.GetoptError:
        print('main.py -i <inputfolder> -o <outputfolder>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('main.py -i <inputfolder> -o <outputfolder>')
            sys.exit()
        elif opt in ("-i", "--inputfolder"):
            inputfolder = arg
        elif opt in ("-o", "--outputfolder"):
            outputfolder = arg
    print('Input Folder is {}'.format(inputfolder))
    print('Output Folder is {}'.format(outputfolder))
    if inputfolder != "" and outputfolder != "":
        print('Run programming')
        #input_folder = r"ShowCase01Test-Input-29059811"
        #output_folder = r"ShowCase01Test-Output-29059811"
        input_folder = inputfolder
        output_folder = outputfolder
        generate_rs_rp_by_ct_folder(
            input_ct_folder=input_folder,
            output_rs_rp_folder=output_folder,
            model_name="MRCNN_Brachy")
    else:
        print("Do nothing because you don't set some of input folder or output folder. ")
        print("You may try the following command ")
        print("$ python main.py -i ShowCase01Test-Input-29059811 -o ShowCase01Test-Output-29059811")

if __name__ == "__main__":
    dev_test_code_running()
    #main(sys.argv[1:])

    exit()


