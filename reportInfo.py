# -*- coding: UTF-8 -*-

import datetime

class ReportInfo:
    def __init__(self, starttime, total_files):
        self.starttime = starttime
        self.file_start_time = datetime.datetime.now()
        self.align_start_time = datetime.datetime.now()
        self.align_estimated_time = datetime.datetime.now()
        self.input_file = ''
        self.total_files = total_files
        self.files_left = total_files
        self.current_path_knot = 1
        self.total_path_knots = 0
        self.totalestimatedtime = datetime.datetime.now()
        self.file_elapsed_time = datetime.datetime.now()
        self.align_elapsed_time = datetime.datetime.now()
        self.gale_church_elapsed_time = datetime.datetime.now()
        self.calc_anchors_elapsed_time = datetime.datetime.now()
        self.calc_elapsed_labse = datetime.datetime.now()
        self.greedy_algorithm_elapsed_time = datetime.datetime.now()
        self.file_processing_stage = 'Initializing '
        self.source_file_length = 0
        self.target_file_length = 0
        self.anchors = []

    def print_info(self):
        text_out = 'File: ' + str(self.input_file) + '\n' + 'File Elapsed Time: ' + str(self.file_elapsed_time) + '\n' + \
            'Source file length: ' + str(self.source_file_length) + '\n' + \
            'Target file length: ' + str(self.target_file_length) + '\n' + 'Nodes: ' + str(self.total_path_knots) + '\n' + \
            'Gale-Church Elapsed Time: ' + str(self.gale_church_elapsed_time) + '\n' + 'Align Elapsed Time: ' + str(self.align_elapsed_time) + '\n' + \
            'Calc Anchors Elapsed Time: ' + str(self.calc_anchors_elapsed_time) + '\n' + 'Greedy Algorithm Elapsed Time: ' + str(self.greedy_algorithm_elapsed_time) + '\n' + \
            'Anchors: ' + str(self.anchors) + '\n'
        return text_out

    def init_file(self, input_file):
        self.input_file = input_file
        self.file_processing_stage = 'Initializing...'
        self.file_start_time = datetime.datetime.now()

    def set_file(self, source_file_length, target_file_length):
        self.source_file_length = source_file_length
        self.target_file_length = target_file_length
        self.file_processing_stage = 'Initializing '

    def set_status(self, status):
        self.file_processing_stage = status

    def set_aligning(self, total_path_knots):
        self.total_path_knots = total_path_knots
        self.align_start_time = datetime.datetime.now()
        self.file_processing_stage = 'Aligning '
        self.files_left = self.files_left - 1

    def set_anchoring(self):
        self.file_processing_stage = 'Anchoring '

    def set_elapsed_gale_church(self, galechurch):
        self.gale_church_elapsed_time = galechurch


    def set_elapsed_calc_labse(self, calclabse):
        self.calc_elapsed_labse = calclabse


    def set_elapsed_calc_anchors(self, calctime):
        self.calc_anchors_elapsed_time = calctime

    def set_total_calculations(self, total_calculations):
        self.total_path_knots = total_calculations

    def add_nodes(self, calculated_nodes):
        self.current_path_knot += calculated_nodes

    def set_elapsed_greedy(self, greedytime):
        self.greedy_algorithm_elapsed_time = greedytime


    def set_elapsed_align(self, aligntime):
        self.align_elapsed_time = aligntime

    def set_anchors(self, anchors):
        self.anchors = anchors

    def update_aligning(self, current_path_knot):
        self.current_path_knot = current_path_knot

    def update_times(self):
        self.file_elapsed_time = datetime.datetime.now() - self.file_start_time
        self.align_elapsed_time = datetime.datetime.now() - self.align_start_time
        self.align_estimated_time = (self.align_elapsed_time * (self.total_path_knots / self.current_path_knot)) + (self.align_start_time - self.file_start_time)
        self.current_path_knot = self.current_path_knot
        self.total_path_knots = self.total_path_knots
        self.file_processing_stage = self.file_processing_stage
        self.source_file_length = self.source_file_length
        self.target_file_length = self.target_file_length
