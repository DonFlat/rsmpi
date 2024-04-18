#![allow(missing_docs)]

use std::ffi::{c_double, c_int, c_void};
use std::mem::{ManuallyDrop, size_of};
use std::ptr;
use mpi_sys::{MPI_Aint, MPI_Win, RSMPI_COMM_WORLD, RSMPI_DOUBLE, RSMPI_INFO_NULL};
use crate::ffi;

pub struct CreatedWindow {
    pub window_vector: Vec<f64>,
    pub window_ptr: MPI_Win
}

impl CreatedWindow {
    pub fn new(size: usize) -> Self {
        let mut win = Self {
            window_vector: vec![0f64; size],
            window_ptr: ptr::null_mut()
        };
        unsafe {
            ffi::MPI_Win_create(
                win.window_vector.as_mut_ptr() as *mut c_void,
                (size * size_of::<c_double>()) as MPI_Aint,
                size_of::<c_double>() as c_int,
                RSMPI_INFO_NULL,
                RSMPI_COMM_WORLD,
                &mut win.window_ptr
            );
        }
        return win;
    }
}

pub struct AllocatedWindow {
    pub window_vector: ManuallyDrop<Vec<f64>>,
    pub window_ptr: MPI_Win
}

impl AllocatedWindow {
    pub fn new(size: usize) -> Self {
        let mut window_base: *mut f64 = ptr::null_mut();
        let mut window_handle: MPI_Win = ptr::null_mut();
        unsafe {
            ffi::MPI_Win_allocate(
                (size * size_of::<c_double>()) as MPI_Aint,
                size_of::<c_double>() as c_int,
                RSMPI_INFO_NULL,
                RSMPI_COMM_WORLD,
                &mut window_base as *mut *mut _ as *mut c_void,
                &mut window_handle
            );
            let win = Self {
                window_vector: ManuallyDrop::new(unsafe { Vec::from_raw_parts(window_base, size, size) }),
                window_ptr: window_handle
            };
            return win;
        }
    }
}

trait WindowOperations {
    fn get_whole_vector(&mut self, target_rank: usize);
    fn put_whole_vector(&mut self, target_rank: usize);
    fn fence(&self);
}

impl WindowOperations for CreatedWindow {
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

impl WindowOperations for AllocatedWindow {
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

fn get(vec: &mut Vec<f64>, target_rank: usize, window: MPI_Win) {
    unsafe {
        ffi::MPI_Get(
            vec.as_mut_ptr() as *mut c_void,
            vec.len() as c_int,
            RSMPI_DOUBLE,
            target_rank as c_int,
            0,
            vec.len() as c_int,
            RSMPI_DOUBLE,
            window
        );
    }
}
fn put(vec: &mut Vec<f64>, target_rank: usize, window: MPI_Win) {
    unsafe {
        ffi::MPI_Put(
            vec.as_mut_ptr() as *mut c_void,
            vec.len() as c_int,
            RSMPI_DOUBLE,
            target_rank as c_int,
            0,
            vec.len() as c_int,
            RSMPI_DOUBLE,
            window
        );
    }
}

fn fence(window: MPI_Win) {
    unsafe {
        ffi::MPI_Win_fence(0, window);
    }
}
