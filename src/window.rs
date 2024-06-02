#![allow(missing_docs)]

use std::ffi::{c_int, c_void};
use std::mem::{ManuallyDrop};
use mpi_sys::{MPI_Aint, MPI_Put, MPI_Win, RSMPI_COMM_WORLD, RSMPI_INFO_NULL};
use crate::ffi;
use crate::traits::{AsRaw, Equivalence};

pub struct CreatedWindow<'a, T> where T: Equivalence {
    pub window_vec_ptr: &'a mut Vec<T>,
    pub window_base_ptr: MPI_Win
}

pub struct AllocatedWindow<T> where T: Equivalence {
    pub window_vector: ManuallyDrop<Vec<T>>,
    pub window_ptr: MPI_Win
}


impl<'a, T> WindowOperations<T> for CreatedWindow<'a, T> where T: Equivalence {
    fn get_whole_vector(&mut self, target_rank: usize) {
        get(self.window_vec_ptr, target_rank, self.window_base_ptr);
    }

    fn put(&mut self, origin_disp: usize, origin_count: usize, target_rank: usize, target_disp: usize, target_count: usize) {
        unsafe {
            MPI_Put(
                self.window_vec_ptr.as_ptr().add(origin_disp) as *mut c_void,
                origin_count as c_int,
                T::equivalent_datatype().as_raw(),
                target_rank as c_int,
                target_disp as MPI_Aint,
                target_count as c_int,
                T::equivalent_datatype().as_raw(),
                self.window_base_ptr
            );
        }
    }

    fn put_whole_vector(&mut self, target_rank: usize) {
        put(self.window_vec_ptr, target_rank, self.window_base_ptr);
    }
    fn fence(&self) {
        fence(self.window_base_ptr);
    }
}

pub trait WindowOperations <T> {
    fn get_whole_vector(&mut self, target_rank: usize);
    fn put(
        &mut self,
        origin_disp: usize,
        origin_count: usize,
        target_rank: usize,
        target_disp: usize,
        target_count: usize
    );
    fn put_whole_vector(&mut self, target_rank: usize);
    fn fence(&self);
}


impl<T> WindowOperations<T> for AllocatedWindow<T> where T: Equivalence {
    fn get_whole_vector(&mut self, target_rank: usize) {
        get(&mut self.window_vector, target_rank, self.window_ptr);
    }

    fn put(&mut self, origin_disp: usize, origin_count: usize, target_rank: usize, target_disp: usize, target_count: usize) {
        unsafe {
            MPI_Put(
                self.window_vector.as_ptr().add(origin_disp) as *mut c_void,
                origin_count as c_int,
                T::equivalent_datatype().as_raw(),
                target_rank as c_int,
                target_disp as MPI_Aint,
                target_count as c_int,
                T::equivalent_datatype().as_raw(),
                self.window_ptr
            );
        }
    }

    fn put_whole_vector(&mut self, target_rank: usize) {
        put(&mut self.window_vector, target_rank, self.window_ptr);
    }
    fn fence(&self) {
        fence(self.window_ptr);
    }
}

pub fn get<T>(vec: &mut Vec<T>, target_rank: usize, window: MPI_Win) where T: Equivalence {
    unsafe {
        ffi::MPI_Get(
            vec.as_mut_ptr() as *mut c_void,
            vec.len() as c_int,
            T::equivalent_datatype().as_raw(),
            target_rank as c_int,
            0,
            vec.len() as c_int,
            T::equivalent_datatype().as_raw(),
            window
        );
    }
}
pub fn put<T>(vec: &mut Vec<T>, target_rank: usize, window: MPI_Win) where T: Equivalence {
    unsafe {
        ffi::MPI_Put(
            vec.as_mut_ptr() as *mut c_void,
            vec.len() as c_int,
            T::equivalent_datatype().as_raw(),
            target_rank as c_int,
            0,
            vec.len() as c_int,
            T::equivalent_datatype().as_raw(),
            window
        );
    }
}

fn fence(window: MPI_Win) {
    unsafe {
        ffi::MPI_Win_fence(0, window);
    }
}

impl<T> Drop for AllocatedWindow<T> where T: Equivalence {
    fn drop(&mut self) {
        unsafe {
            ffi::MPI_Win_free(&mut self.window_ptr);
        }
    }
}

impl<'a, T> Drop for CreatedWindow<'a, T> where T: Equivalence {
    fn drop(&mut self) {
        unsafe {
            ffi::MPI_Win_free(&mut self.window_base_ptr);
        }
    }
}