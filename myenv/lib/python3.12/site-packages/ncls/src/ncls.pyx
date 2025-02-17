# import numpy as cnp
import sys

cimport cython
from libc.stdint cimport int64_t
from libc.stdlib cimport malloc

cimport ncls.src.cncls as cn

import numpy as np


cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b
# import ctypes as c

try:
    dummy = profile
except:
    profile = lambda x: x

cdef class NCLS64:

    cdef cn.SublistHeader *subheader
    cdef cn.IntervalMap *im
    cdef int n, ntop
    cdef int nlists

    # build NCLS from array of starts, ends, values
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    def __cinit__(self, const int64_t [::1] starts=None, const int64_t [::1] ends=None, const int64_t [::1] ids=None):

        if None in (starts, ends, ids):
            return

        if len(starts) == 0 or len(ends) == 0 or len(ids) == 0:
            return

        cdef int i
        cdef length = len(starts)
        self.close() # DUMP OUR EXISTING MEMORY
        self.n = len(starts)
        self.im = cn.interval_map_alloc(self.n)
        if self.im == NULL:
            raise MemoryError('unable to allocate IntervalMap[%d]' % self.n)
        i = 0
        for i in range(length):
            self.im[i].start = starts[i]
            self.im[i].end = ends[i]
            self.im[i].target_id = <long> ids[i]
            self.im[i].sublist = -1

        self.subheader = cn.build_nested_list(self.im, self.n, &(self.ntop), &(self.nlists))

    def intervals(self, take=None):
        intervals = []

        cdef int i = 0
        cdef int _take = int(take) if not take is None else len(self)
        if not self.im: # if empty
            return []

        for i in range(_take):
            intervals.append((self.im[i].start, self.im[i].end, self.im[i].target_id))

        return intervals


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef all_overlaps_both(self,
                            const int64_t [::1] starts,
                            const int64_t [::1] ends,
                            const int64_t [::1] indexes):

        cdef int i = 0
        cdef int nhit = 0
        cdef int length = len(starts)
        cdef int loop_counter = 0
        cdef int nfound = 0
        cdef int spent = 0

        output_arr = np.zeros(length, dtype=np.int64)
        output_arr_other = np.zeros(length, dtype=np.int64)
        cdef int64_t [::1] output
        cdef int64_t [::1] output_other

        output = output_arr
        output_other = output_arr_other

        cdef cn.IntervalIterator *it
        cdef cn.IntervalIterator *it_alloc

        cdef cn.IntervalMap im_buf[1024]
        if not self.im: # if empty
            return [], []

        it_alloc = cn.interval_iterator_alloc()
        it = it_alloc
        for loop_counter in range(length):

            spent = 0
            while not spent:
                i = 0
                cn.find_intervals(it, starts[loop_counter], ends[loop_counter], self.im, self.ntop,
                                self.subheader, self.nlists, im_buf, 1024,
                                &(nhit), &(it)) # GET NEXT BUFFER CHUNK

                if nfound + nhit >= length:

                    length = (length + nhit) * 2
                    output_arr = np.resize(output_arr, length)
                    output_arr_other = np.resize(output_arr_other, length)
                    output = output_arr
                    output_other = output_arr_other

                while i < nhit:
                    output[nfound] = indexes[loop_counter]
                    output_other[nfound] = im_buf[i].target_id

                    # print("  output[nfound]", output[nfound])
                    # print("  output_other[nfound]", output_other[nfound])

                    nfound += 1
                    i += 1

                if nhit < 1024:
                    spent = 1

            cn.reset_interval_iterator(it_alloc)
            it = it_alloc

        cn.free_interval_iterator(it_alloc)

        return output_arr[:nfound], output_arr_other[:nfound]


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef last_overlap_both(self, const int64_t [::1] starts, const int64_t [::1] ends, const int64_t [::1] indexes):

        cdef int i = 0
        cdef int nhit = 0
        cdef int length = len(starts)
        cdef int loop_counter = 0
        cdef int nfound = 0
        cdef int max_end = -1
        cdef int spent = 0

        output_arr = np.zeros(length, dtype=np.int64)
        output_arr_other = np.zeros(length, dtype=np.int64)
        cdef int64_t [::1] output
        cdef int64_t [::1] output_other

        output = output_arr
        output_other = output_arr_other

        cdef cn.IntervalIterator *it
        cdef cn.IntervalIterator *it_alloc

        cdef cn.IntervalMap im_buf[1024]
        if not self.im: # if empty
            return [], []

        it_alloc = cn.interval_iterator_alloc()
        it = it_alloc
        for loop_counter in range(length):
            max_end = -1
            spent = 0
            while not spent:
                i = 0
                cn.find_intervals(it, starts[loop_counter], ends[loop_counter], self.im, self.ntop,
                                self.subheader, self.nlists, im_buf, 1024,
                                &(nhit), &(it)) # GET NEXT BUFFER CHUNK
                if nhit:
                    while i < nhit:
                        if im_buf[i].end >= max_end:
                            # print("max_end", im_buf[i].end)
                            output[nfound] = indexes[loop_counter]
                            output_other[nfound] = im_buf[i].target_id
                            max_end = im_buf[i].end

                        i += 1

                    nfound += 1

                if nhit < 1024:
                    spent = 1

            cn.reset_interval_iterator(it_alloc)
            it = it_alloc

        cn.free_interval_iterator(it_alloc)

        return output_arr[:nfound], output_arr_other[:nfound]


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef k_overlaps_both(self, const int64_t [::1] starts, const int64_t [::1] ends, const int64_t [::1] indexes, int k):

        cdef int i = 0
        cdef int nhit = 0
        cdef int length = len(starts)
        cdef int loop_counter = 0
        cdef int nfound = 0
        cdef int spent = 0

        output_arr = np.zeros(length, dtype=np.int64)
        output_arr_other = np.zeros(length, dtype=np.int64)
        cdef int64_t [::1] output
        cdef int64_t [::1] output_other

        output = output_arr
        output_other = output_arr_other

        cdef cn.IntervalIterator *it
        cdef cn.IntervalIterator *it_alloc

        cdef cn.IntervalMap im_buf[1024]
        if not self.im: # if empty
            return [], []

        it_alloc = cn.interval_iterator_alloc()
        it = it_alloc
        for loop_counter in range(length):

            # remember first pointer for dealloc
            spent = 0
            while not spent:
                i = 0
                cn.find_intervals(it, starts[loop_counter], ends[loop_counter], self.im, self.ntop,
                                self.subheader, self.nlists, im_buf, 1024,
                                &(nhit), &(it)) # GET NEXT BUFFER CHUNK

                if nfound + nhit >= length:

                    length = (length + nhit) * 2
                    output_arr = np.resize(output_arr, length)
                    output_arr_other = np.resize(output_arr_other, length)
                    output = output_arr
                    output_other = output_arr_other

                if k < nhit:
                    nhit = k

                while i < nhit:

                    output[nfound] = indexes[loop_counter]
                    output_other[nfound] = im_buf[i].target_id

                    nfound += 1
                    i += 1

                if nhit < 1024:
                    spent = 1

            cn.reset_interval_iterator(it_alloc)
            it = it_alloc

        cn.free_interval_iterator(it_alloc)

        return output_arr[:nfound], output_arr_other[:nfound]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef all_overlaps_self(self, const int64_t [::1] starts, const int64_t [::1] ends, const int64_t [::1] indexes):

        cdef int i
        cdef int nhit = 0
        cdef int length = len(starts)
        cdef int loop_counter = 0
        cdef int nfound = 0
        cdef int spent = 0

        output_arr = np.zeros(length, dtype=np.int64)
        cdef int64_t [::1] output

        output = output_arr

        cdef cn.IntervalIterator *it
        cdef cn.IntervalIterator *it_alloc

        cdef cn.IntervalMap im_buf[1024]
        if not self.im: # if empty
            return [], []

        it_alloc = cn.interval_iterator_alloc()
        it = it_alloc
        for loop_counter in range(length):

            # remember first pointer for dealloc

            spent = 0
            while not spent:
                i = 0
                cn.find_intervals(it, starts[loop_counter], ends[loop_counter], self.im, self.ntop,
                                self.subheader, self.nlists, im_buf, 1024,
                                &(nhit), &(it)) # GET NEXT BUFFER CHUNK

                if nfound + nhit >= length:

                    length = (length + nhit) * 2
                    output_arr = np.resize(output_arr, length)
                    output = output_arr

                for i in range(nhit):

                    output[nfound] = indexes[loop_counter]

                    nfound += 1

                if nhit < 1024:
                    spent = 1

            cn.reset_interval_iterator(it_alloc)
            it = it_alloc

        cn.free_interval_iterator(it_alloc)

        return output_arr[:nfound]


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef coverage(self, const int64_t [::1] starts, const int64_t [::1] ends, const int64_t [::1] indexes):

        # assumes the ncls to not contain any overlapping intervals

        cdef int i = 0
        cdef int64_t start = 0
        cdef int64_t end = 0
        cdef int other_start = 0
        cdef int other_end = 0
        cdef int nhit = 0
        cdef int length = len(starts)
        cdef int loop_counter = 0
        cdef int nfound = 0
        cdef int spent = 0

        output_arr_length = np.zeros(length, dtype=np.int64)
        cdef int64_t [::1] output_length

        output_length = output_arr_length

        cdef cn.IntervalIterator *it
        cdef cn.IntervalIterator *it_alloc

        cdef cn.IntervalMap im_buf[1024]
        if not self.im: # if empty
            return [], []

        it_alloc = cn.interval_iterator_alloc()
        it = it_alloc
        for loop_counter in range(length):

            start = starts[loop_counter]
            end = ends[loop_counter]
            # remember first pointer for dealloc
            spent = 0
            while not spent:
                i = 0
                cn.find_intervals(it, starts[loop_counter], ends[loop_counter], self.im, self.ntop,
                                self.subheader, self.nlists, im_buf, 1024,
                                &(nhit), &(it)) # GET NEXT BUFFER CHUNK

                while i < nhit:
                    output_length[loop_counter] += int_min(im_buf[i].end, end) - int_max(im_buf[i].start, start)
                    i += 1

                if nhit < 1024:
                    spent = 1

            cn.reset_interval_iterator(it_alloc)
            it = it_alloc

        cn.free_interval_iterator(it_alloc)

        return output_arr_length

    def __len__(self):
        return self.n

    def __str__(self):

        contents = ["Number intervals:", self.n, "Number of intervals in main list:", self.ntop, "Number of intervals with subintervals:", self.nlists, "Percentage in top-level interval", self.ntop/float(self.n)]
        return "NCLS64\n------\n" + "\n".join(str(c) for c in contents)

    def __dealloc__(self):
        'remember: dealloc cannot call other methods!'
        if self.subheader:
            cn.free(self.subheader)
        if self.im:
            cn.free(self.im)

    def close(self):
        if self.subheader:
            cn.free(self.subheader)
        if self.im:
            cn.free(self.im)
            self.subheader = NULL
            self.im = NULL

        return None

    def find_overlap(self, int64_t start, int64_t end):
        if not self.im: # RAISE EXCEPTION IF NO DATA
            return []

        return NCLSIterator(start, end, self)


    cpdef has_overlap(self, int64_t start, int64_t end):
        cdef int nhit = 0

        cdef cn.IntervalIterator *it
        cdef cn.IntervalMap im_buf[1024]
        if not self.im: # if empty
            return 0

        it = cn.interval_iterator_alloc()

        while it:
            cn.find_intervals(it, start, end, self.im, self.ntop,
                              self.subheader, self.nlists, im_buf, 1024,
                              &(nhit), &(it)) # GET NEXT BUFFER CHUNK

            if nhit > 0:
                cn.free_interval_iterator(it)
                return True

        cn.free_interval_iterator(it)

        return False



    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef set_difference_helper(self, const int64_t [::1] starts, const int64_t [::1] ends, const int64_t [::1] indexes,
                                const int64_t [::1] nhits):

        cdef int i
        cdef int nhit = 0
        cdef int nfound = 0
        cdef int64_t nstart = 0
        cdef int64_t nend = 0
        cdef int length = len(starts)
        cdef int loop_counter = 0
        cdef int overlap_type_nb = 0
        cdef int na = -1
        cdef int spent = 0


        output_arr = np.zeros(length, dtype=np.int64)
        output_arr_start = np.zeros(length, dtype=np.int64)
        output_arr_end = np.zeros(length, dtype=np.int64)
        cdef int64_t [::1] output
        cdef int64_t [::1] output_start
        cdef int64_t [::1] output_end

        output = output_arr
        output_start = output_arr_start
        output_end = output_arr_end

        cdef cn.IntervalIterator *it
        cdef cn.IntervalIterator *it_alloc
        cdef cn.IntervalMap im_buf[1024]
        if not self.im: # if empty
            return [], [], []

        it_alloc = cn.interval_iterator_alloc()
        it = it_alloc
        for loop_counter in range(length):

            nhit = nhits[loop_counter]
            nstart = starts[loop_counter]
            nend = ends[loop_counter]

            while nhit > 0:
                i = 0
                cn.find_intervals(it, starts[loop_counter], ends[loop_counter], self.im, self.ntop,
                                self.subheader, self.nlists, im_buf, 1024,
                                &(na), &(it)) # GET NEXT BUFFER CHUNK

                if nfound + nhit >= length:
                    length = (length + nhit) * 2
                    output_arr = np.resize(output_arr, length)
                    output_arr_start = np.resize(output_arr_start, length)
                    output = output_arr
                    output_start = output_arr_start
                    output_arr_end = np.resize(output_arr_end, length)
                    output_end = output_arr_end

                # B covers whole of A; ignore
                if nhit == 1 and starts[loop_counter] > im_buf[i].start and ends[loop_counter] < im_buf[i].end:
                    # print("ignore me!")
                    output_start[nfound] = -1
                    output_end[nfound] = -1
                    output[nfound] = indexes[loop_counter]
                    i = nhit
                    nfound += 1
                    break

                max_i = 1024 if nhit > 1024 else nhit

                while i < max_i:
                    # in case the start contributes nothing
                    if nstart < im_buf[i].start:
                        output[nfound] = indexes[loop_counter]
                        output_start[nfound] = nstart
                        output_end[nfound] = im_buf[i].start
                        nfound += 1
                    nstart = im_buf[i].end

                    i += 1

                nhit = nhit - 1024

                if nhit <= 0:
                    i = i - 1
                    if im_buf[i].start <= nstart and im_buf[i].end >= ends[loop_counter]:
                        # print("im_buf[i].start <= nstart and im_buf[i].end >= ends[loop_counter]")
                        #print("we are here " * 10)

                        output_start[nfound] = -1
                        output_end[nfound] = -1
                        output[nfound] = <long> indexes[loop_counter]
                        nfound += 1
                    else:
                        if im_buf[i].start > nstart:
                            output[nfound] = indexes[loop_counter]
                            output_start[nfound] = nstart
                            output_end[nfound] = im_buf[i].start
                            nfound += 1

                        if im_buf[i].end < ends[loop_counter]:
                            output[nfound] = indexes[loop_counter]
                            output_start[nfound] = im_buf[i].end
                            output_end[nfound] = ends[loop_counter]
                            nfound += 1

            cn.reset_interval_iterator(it_alloc)
            it = it_alloc

        cn.free_interval_iterator(it_alloc)

        return output_arr[:nfound], output_arr_start[:nfound], output_arr_end[:nfound]


    # this one is actually slower!!!!
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef first_overlap_both(self, const int64_t [::1] starts, const int64_t [::1] ends, const int64_t [::1] indexes):

        cdef int ix = 0
        cdef int length = len(starts)
        cdef int loop_counter = 0
        cdef int nfound = 0

        output_arr = np.zeros(length, dtype=np.int64)
        output_arr_other = np.zeros(length, dtype=np.int64)
        cdef int64_t [::1] output
        cdef int64_t [::1] output_other

        output = output_arr
        output_other = output_arr_other

        cdef cn.IntervalIterator *it
        cdef cn.IntervalIterator *it_alloc

        cdef cn.IntervalMap im_buf[1024]
        if not self.im: # if empty
            return [], []


        for loop_counter in range(length):

            # remember first pointer for dealloc
            ix = cn.find_overlap_start(starts[loop_counter], ends[loop_counter], self.im, self.ntop)

            if ix != -1:
                output[nfound] = indexes[loop_counter]
                output_other[nfound] = self.im[ix].target_id

                nfound += 1

        return output_arr[:nfound], output_arr_other[:nfound]


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef all_containments_both(self, const int64_t [::1] starts, const int64_t [::1] ends, const int64_t [::1] indexes):

        cdef int i
        cdef int nhit = 0
        cdef int length = len(starts)
        cdef int loop_counter = 0
        cdef int nfound = 0
        cdef int64_t start, end
        cdef int spent = 0

        output_arr = np.zeros(length, dtype=np.int64)
        output_arr_other = np.zeros(length, dtype=np.int64)
        cdef int64_t [::1] output
        cdef int64_t [::1] output_other

        output = output_arr
        output_other = output_arr_other

        cdef cn.IntervalIterator *it
        cdef cn.IntervalIterator *it_alloc


        cdef cn.IntervalMap im_buf[1024]
        if not self.im: # if empty
            return [], []


        it_alloc = cn.interval_iterator_alloc()
        it = it_alloc

        for loop_counter in range(length):

            start = starts[loop_counter]
            end = ends[loop_counter]
            spent = 0
            while not spent:
                i = 0
                cn.find_intervals(it, start, end, self.im, self.ntop,
                                self.subheader, self.nlists, im_buf, 1024,
                                &(nhit), &(it)) # GET NEXT BUFFER CHUNK

                while i < nhit:

                    if im_buf[i].start <= start and im_buf[i].end >= end:

                        if nfound >= length:

                            length = nfound * 2
                            output_arr = np.resize(output_arr, length)
                            output_arr_other = np.resize(output_arr_other, length)
                            output = output_arr
                            output_other = output_arr_other

                        output[nfound] = indexes[loop_counter]
                        output_other[nfound] = im_buf[i].target_id

                        nfound += 1
                    i += 1

                if nhit < 1024:
                    spent = 1

            cn.reset_interval_iterator(it_alloc)
            it = it_alloc

        cn.free_interval_iterator(it_alloc)

        return output_arr[:nfound], output_arr_other[:nfound]




    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef has_overlaps(self, const int64_t [::1] starts, const int64_t [::1] ends, const int64_t [::1] indexes):

        cdef int i = 0
        cdef int ix = 0
        cdef int length = len(starts)
        cdef int nfound = 0

        if not self.im: # if empty
            return []

        output_arr = np.zeros(length, dtype=np.int64)
        cdef int64_t [::1] output
        output = output_arr

        for i in range(length):

            ix = cn.find_overlap_start(starts[i], ends[i], self.im, self.ntop)

            if ix != -1:
                output[nfound] = indexes[i]
                nfound += 1

            i += 1

        return output_arr[:nfound]



cdef class NCLSIterator:

    cdef cn.IntervalIterator *it
    cdef cn.IntervalIterator *it_alloc
    cdef cn.IntervalMap im_buf[1024]
    cdef int nhit, ihit
    cdef int64_t start, end
    cdef NCLS64 db

    def __cinit__(self, int64_t start, int64_t end, NCLS64 db not None):
        self.it = cn.interval_iterator_alloc()
        self.it_alloc = self.it
        self.start = start
        self.end = end
        self.db = db
        self.nhit = 0
        self.ihit = 0


    def __iter__(self):
        return self


    cdef int cnext(self): # c VERSION OF ITERATOR next METHOD RETURNS INDEX
        cdef int64_t i
        if self.ihit >= self.nhit: # TRY TO GET ONE MORE BUFFER CHUNK OF HITS
            if self.it == NULL: # ITERATOR IS EXHAUSTED
                return -1
            cn.find_intervals(self.it, self.start, self.end, self.db.im, self.db.ntop,
                           self.db.subheader, self.db.nlists, self.im_buf, 1024,
                           &(self.nhit), &(self.it)) # GET NEXT BUFFER CHUNK
            self.ihit = 0 # START ITERATING FROM START OF BUFFER

        if self.ihit < self.nhit: # RETURN NEXT ITEM FROM BUFFER
            i = self.ihit
            self.ihit = self.ihit + 1 # ADVANCE THE BUFFER COUNTER
            return i
        else: # BUFFER WAS EMPTY, NO HITS TO ITERATE OVER...
            return -1


    # PYTHON VERSION OF next RETURNS HIT AS A TUPLE
    def __next__(self): # PYREX USES THIS NON-STANDARD NAME INSTEAD OF next()!!!
        cdef int i
        i = self.cnext()
        if i >= 0:
            return (self.im_buf[i].start, self.im_buf[i].end, self.im_buf[i].target_id)
        else:
            raise StopIteration

    def __dealloc__(self):
        'remember: dealloc cannot call other methods!'
        cn.free_interval_iterator(self.it_alloc)


    def find_overlap(self, int64_t start, int64_t end):
        if not self.im:
            return []

        return NCLSIterator(start, end, self)
