#![allow(missing_docs)]

use std::ffi::{c_double, c_int, c_void};
use std::mem::size_of;
use std::ptr;
use mpi_sys::{MPI_Aint, MPI_Win, RSMPI_COMM_WORLD, RSMPI_DOUBLE, RSMPI_INFO_NULL};
use crate::ffi;

pub struct Window {
    pub window_vector: Vec<f64>,
    pub window_ptr: MPI_Win
}

impl Window {
    pub fn new(size: usize) -> Window {
        let mut win = Window {
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

    pub fn get_whole_vector(&mut self, target_rank: usize) {
        unsafe {
            ffi::MPI_Get(
                self.window_vector.as_mut_ptr() as *mut c_void,
                self.window_vector.len() as c_int,
                RSMPI_DOUBLE,
                target_rank as c_int,
                0,
                self.window_vector.len() as c_int,
                RSMPI_DOUBLE,
                self.window_ptr
            );
        }
    }

    pub fn put_whole_vector(&mut self, target_rank: usize) {
        unsafe {
            ffi::MPI_Put(
                self.window_vector.as_mut_ptr() as *mut c_void,
                self.window_vector.len() as c_int,
                RSMPI_DOUBLE,
                target_rank as c_int,
                0,
                self.window_vector.len() as c_int,
                RSMPI_DOUBLE,
                self.window_ptr
            );
        }
    }

    pub fn fence(&self) {
        unsafe {
            ffi::MPI_Win_fence(0, self.window_ptr);
        }
    }
}

impl Drop for Window {
    fn drop(&mut self) {
        println!("Drop is called");
        unsafe {
            ffi::MPI_Win_free(&mut self.window_ptr);
        }
    }
}