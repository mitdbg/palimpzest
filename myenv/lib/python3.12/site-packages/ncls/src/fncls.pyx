cimport cython
from libc.stdint cimport int64_t
from libc.stdlib cimport malloc

cimport ncls.src.cfncls as cn
cimport ncls.src.cncls as cn

import numpy as np


cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b
# import ctypes as c

try:
    dummy = profile
except:
    profile = lambda x: x


cdef class FNCLS:

    cdef cn.SublistHeader *subheader
    cdef cn.IntervalMap *im
    cdef int n, ntop
    cdef int nlists

    # build NCLS from array of starts, ends, values
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    def __cinit__(self, const double [::1] starts=None, const double [::1] ends=None, const int64_t [::1] ids=None):

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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef all_overlaps_both(self, const double [::1] starts, const double [::1] ends, const int64_t [::1] indexes):

        cdef int i = 0
        cdef int nhit = 0
        cdef int length = len(starts)
        cdef int loop_counter = 0
        cdef int nfound = 0

        output_arr = np.zeros(length, dtype=np.double)
        output_arr_other = np.zeros(length, dtype=np.double)
        cdef double [::1] output
        cdef double [::1] output_other

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

            # print("---search---: {} {}".format(starts[loop_counter], ends[loop_counter]))
            while it:
                i = 0
                cn.find_intervals(it, starts[loop_counter], ends[loop_counter], self.im, self.ntop,
                                self.subheader, self.nlists, im_buf, 1024,
                                &(nhit), &(it)) # GET NEXT BUFFER CHUNK

                # print("nhit", nhit)
                if nfound + nhit >= length:

                    length = (length + nhit) * 2
                    output_arr = np.resize(output_arr, length)
                    output_arr_other = np.resize(output_arr_other, length)
                    output = output_arr
                    output_other = output_arr_other

                while i < nhit:
                    # print("  i", i)

                    # print("length", length)
                    # print("nfound", nfound)
                    # print("loop_counter", loop_counter)
                    output[nfound] = indexes[loop_counter]
                    output_other[nfound] = im_buf[i].target_id

                    # print("  hit {} {}".format(im_buf[i].start, im_buf[i].end))
                    # print("  output[nfound]", output[nfound])
                    # print("  output_other[nfound]", output_other[nfound])

                    nfound += 1
                    i += 1

            cn.reset_interval_iterator(it_alloc)
            it = it_alloc

        cn.free_interval_iterator(it_alloc)
        # end = time()

        # print("ncls time:", end - start)

        return output_arr[:nfound], output_arr_other[:nfound]


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
