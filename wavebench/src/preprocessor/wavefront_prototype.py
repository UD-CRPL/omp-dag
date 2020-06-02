#!/usr/bin/env python

'''
 Robert Searles
 Department of Computer and Information Sciences
 University of Delaware

 wavefront_prototype.py
 Prototype of a wavefront OpenACC pragma. 
 Implemented as a source-to-source preprocessor.
 The result can be fed into any OpenACC compiler (such as PGI).
'''

#################################################################
###    Includes
#################################################################

# Needed for system operations
import sys, os, shutil

# Parsing command line arguments
import argparse

# Regular expressions
import re

#################################################################
###    Class/Function Declarations
#################################################################

def read_source(source_code):
    with open(source_code) as f:
        content = f.readlines()
    return content

def parse_source(source_strs, processed_source, starting_index):
    wavefront_region=[]
    line_index = starting_index

    for idx in xrange(starting_index, len(source_strs)):
        # Split
        curr_line = source_strs[idx].split()

        # Check for wavefront pragma
        if wavefront_check(curr_line) is True:
            # If found, build wavefront region & transform
            curr_idx = build_wavefront_region(source_strs, wavefront_region, idx)

            # Wavefront Transform
            wavefront_region = wavefront_transform(wavefront_region)

            # Add transformed lines to processed_source
            for line in wavefront_region:
                processed_source.append(line)

            # Continue parsing
            parse_source(source_strs, processed_source, curr_idx)
            break

        # Else, add line to processed source
        else:
            processed_source.append(source_strs[idx])

    return

def wavefront_check(line):
    if len(line) >= 3:
        if line[0] == "#pragma" and line[1] == "acc" and line[2][:9] == "wavefront":
            return True
    return False

def build_wavefront_region(source_strs, wavefront_region, idx):
    # Add first line
    wavefront_region.append(source_strs[idx])
    idx+=1

    # Check for { second line
    curr_line = source_strs[idx].split()
    if curr_line[0] == '{':
        wavefront_region.append(source_strs[idx])
        idx+=1
    else:
        print "ERROR: Incorrect wavefront syntax!"
        print "Wavefront pragma must be followed by a region starting with {"
        exit(-1)

    # Add lines until } is reached
    region_complete = False
    open_regions = 1
    while not region_complete:
        curr_line = source_strs[idx].split()
        if curr_line[0] == '}' and open_regions == 1:
            region_complete = True
        elif curr_line[0] == '}':
            open_regions-=1
        if curr_line[0] == '{':
            open_regions+=1
        wavefront_region.append(source_strs[idx])
        idx+=1

    return idx

def wavefront_transform(wavefront_region):
    print '--> Transforming wavefront region'
    transformed_wavefront_region = []
    # Retrieve the dimensions specified by wavefront(<number>)
    wavefront_dims = int(filter(None,re.split('\(|\)|\ ',wavefront_region[0].split()[2]))[1])
    loop_starts = []
    loop_bounds = []
    dim_var_names = []
    idx = 0

    # Gather loop bounds and variables from 'for' loops
    while len(loop_bounds) < wavefront_dims or idx < len(wavefront_region):
        curr_line = wavefront_region[idx].split()
        if curr_line[0] == 'for' and len(curr_line) >= 4:
            # Extract starting bound
            loop_starts.append(filter(None,re.split(';|<|>|=',curr_line[1]))[-1])

            # Extract loop bound
            loop_bounds.append(filter(None,re.split(';|<|>|=',curr_line[2]))[-1])

            # Extract dimension variable name
            dim_var_names.append(filter(None,re.split('\(|=|;',curr_line[1]))[0])
            
        # Increment
        idx+=1

    # print "starts: ", loop_starts, "dims: ", loop_bounds, "\n", "vars: ", dim_var_names

    # If there still aren't enough loop bounds collected, error
    if len(loop_bounds) < wavefront_dims:
        print "ERROR: Wavefront dimension too large!"
        exit(-1)

    # Build transformed wavefront strings

    # num_wavefronts calculation 
    num_wavefronts_str = 'int num_wavefronts = ('
    bounds_idx = 0
    while bounds_idx < len(loop_bounds):
        num_wavefronts_str+=loop_bounds[bounds_idx]
        if bounds_idx < len(loop_bounds)-1:
            num_wavefronts_str+=' + '
        bounds_idx+=1
    num_wavefronts_str+=') - '
    num_wavefronts_str+=str(len(loop_bounds)-1)
    num_wavefronts_str+=';\n\n'
    transformed_wavefront_region.append(num_wavefronts_str)

    # wavefront loop
    transformed_wavefront_region.append('for (int wavefront=0; wavefront < num_wavefronts; wavefront++)\n')

    # Find loop nest
    in_loop_nest = False
    idx = 1
    while not in_loop_nest:
        if wavefront_region[idx].split()[0] == 'for':
            in_loop_nest = True
            break
        else:
            transformed_wavefront_region.append(wavefront_region[idx])
            idx+=1

    # Add gang loop pragma
    gang_loop = '#pragma acc loop independent gang, collapse('
    gang_loop += str(len(loop_bounds)-1)
    gang_loop += ')\n'
    transformed_wavefront_region.append(gang_loop)

    # Skip (delete) outermost loop
    idx+=1

    # Add loop nest
    while in_loop_nest:
        if wavefront_region[idx].split()[0] == '{':
            in_loop_nest = False
        transformed_wavefront_region.append(wavefront_region[idx])
        idx+=1

    # Solve for bounds
    transformed_wavefront_region.append('\n/*--- Solve for outer dim ---*/\n')
    bounds_solve = str(dim_var_names[0])
    bounds_solve += ' = wavefront - ('
    bounds_idx = 1
    while bounds_idx < len(dim_var_names):
        bounds_solve += str(dim_var_names[bounds_idx])
        if bounds_idx < len(dim_var_names)-1:
            bounds_solve += ' + '
        bounds_idx+=1
    bounds_solve += ');\n\n'
    transformed_wavefront_region.append(bounds_solve)

    # Bounds check
    transformed_wavefront_region.append('/*--- Bounds check ---*/\n')
    bounds_check = 'if ('
    bounds_check += dim_var_names[0]
    bounds_check += ' >= '
    bounds_check += loop_starts[0]
    bounds_check += ' && '
    bounds_check += dim_var_names[0]
    bounds_check += ' < '
    bounds_check += loop_bounds[0]
    bounds_check += ')\n'
    transformed_wavefront_region.append(bounds_check)
    transformed_wavefront_region.append('{\n')

    # Add body
    in_body = True
    while in_body:
        if wavefront_region[idx].split()[0] == '}':
            in_body = False
            break
        else:
            transformed_wavefront_region.append(wavefront_region[idx])
            idx+=1

    # Close bounds check region
    transformed_wavefront_region.append('} /*--- Bounds check ---*/\n')

    # Add remainder of wavefront region
    while idx < len(wavefront_region):
        transformed_wavefront_region.append(wavefront_region[idx])
        idx+=1

    return transformed_wavefront_region

def write_output(processed_source, output_file):
    with open(output_file, 'w') as f:
        f.writelines(processed_source)
    return

#################################################################
###    Script Execution
#################################################################

def main():
    print "--> Wavebench Preprocessor"

    print "--> Reading Source Code"
    source = read_source(SOURCE_CODE)
    
    print "--> Parsing Source Code"
    processed_source = []
    parse_source(source, processed_source, 0)

    print "--> Writing Output"
    write_output(processed_source, OUTPUT)

    return
    
if __name__ == "__main__":
    # Used to parse options
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("source_code", help="Source code file that we will operate on.", type=str)
    parser.add_argument("output", help="Output file path/name", type=str)

    # Parse arguments
    args = parser.parse_args()

    # Globals
    SOURCE_CODE = args.source_code
    OUTPUT = args.output

    # execute
    main()
