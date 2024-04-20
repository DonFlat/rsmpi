#![allow(missing_docs)]

use std::ffi::{c_double, c_int, c_void};
use std::mem::{ManuallyDrop, size_of};
use std::ptr;
use mpi_sys::{MPI_Aint, MPI_Win, RSMPI_COMM_WORLD, RSMPI_INFO_NULL};
use crate::ffi;
use crate::traits::{AsRaw, Equivalence};

pub struct CreatedWindow<'a, T> where T: Equivalence {
    pub window_vec_ptr: &'a mut Vec<T>,
    pub window_base_ptr: MPI_Win
}

impl<'a, T> CreatedWindow<'a, T> where T: Equivalence {
    pub fn new(size: usize, vec_ptr: &'a mut Vec<T>) -> Self {
        let mut win = Self {
            window_vec_ptr: vec_ptr,
            window_base_ptr: ptr::null_mut()
        };
        unsafe {
            ffi::MPI_Win_create(
                win.window_vec_ptr.as_mut_ptr() as *mut c_void,
                (size * size_of::<f64>()) as MPI_Aint,
                size_of::<c_double>() as c_int,
                RSMPI_INFO_NULL,
                RSMPI_COMM_WORLD,
                &mut win.window_base_ptr
            );
        }
        return win;
    }
}
impl<'a, T> WindowOperations for CreatedWindow<'a, T> where T: Equivalence {
    fn get_whole_vector(&mut self, target_rank: usize) {
        get(self.window_vec_ptr, target_rank, self.window_base_ptr);
    }
    fn put_whole_vector(&mut self, target_rank: usize) {
        put(self.window_vec_ptr, target_rank, self.window_base_ptr);
    }
    fn fence(&self) {
        fence(self.window_base_ptr);
    }
}

pub struct AllocatedWindow<T> where T: Equivalence {
    pub window_vector: ManuallyDrop<Vec<T>>,
    pub window_ptr: MPI_Win
}

impl<T> AllocatedWindow<T> where T: Equivalence{
    pub fn new(size: usize) -> Self {
        let mut window_base: *mut T = ptr::null_mut();
        let mut window_handle: MPI_Win = ptr::null_mut();
        unsafe {
            ffi::MPI_Win_allocate(
                (size * size_of::<T>()) as MPI_Aint,
                size_of::<T>() as c_int,
                RSMPI_INFO_NULL,
                RSMPI_COMM_WORLD,
                &mut window_base as *mut *mut _ as *mut c_void,
                &mut window_handle
            );
            let win = Self {
                window_vector: ManuallyDrop::new(Vec::from_raw_parts(window_base, size, size)),
                window_ptr: window_handle
            };
            return win;
        }
    }
}

pub trait WindowOperations {
    fn get_whole_vector(&mut self, target_rank: usize);
    fn put_whole_vector(&mut self, target_rank: usize);
    fn fence(&self);
}


impl<T> WindowOperations for AllocatedWindow<T> where T: Equivalence {
    fn get_whole_vector(&mut self, target_rank: usize) {
        get(&mut self.window_vector, target_rank, self.window_ptr);
    }
    fn put_whole_vector(&mut self, target_rank: usize) {
        put(&mut self.window_vector, target_rank, self.window_ptr);
    }
    fn fence(&self) {
        fence(self.window_ptr);
    }
}

fn get<T>(vec: &mut Vec<T>, target_rank: usize, window: MPI_Win) where T: Equivalence {
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
fn put<T>(vec: &mut Vec<T>, target_rank: usize, window: MPI_Win) where T: Equivalence {
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
