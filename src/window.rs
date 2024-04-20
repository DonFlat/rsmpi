#![allow(missing_docs)]

use std::ffi::{c_double, c_int, c_void};
use std::mem::{ManuallyDrop, size_of};
use std::ptr;
use mpi_sys::{MPI_Aint, MPI_Datatype, MPI_Win, RSMPI_COMM_WORLD, RSMPI_DOUBLE, RSMPI_INFO_NULL};
use crate::datatype::{Buffer, Equivalence, SystemDatatype, UserDatatype};
use crate::ffi;
use crate::raw::AsRaw;
use crate::traits::{BufferMut, UncommittedDatatype};

pub struct CreatedWindow<T> where T: Equivalence {
    pub window_vector: Vec<T>,
    pub window_ptr: MPI_Win
}


pub struct AllocatedWindow<T> where T: Equivalence {
    pub window_vector: ManuallyDrop<Vec<T>>,
    pub window_ptr: MPI_Win
}

impl<T> AllocatedWindow<T> where T: Equivalence {
    pub fn new(size: usize) -> Self {
        println!("Now starting to allocate");
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
                window_vector: ManuallyDrop::new(unsafe { Vec::from_raw_parts(window_base, size, size) }),
                window_ptr: window_handle
            };
            println!("Allocated");
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
        println!("Before getting into unsafe");
        unsafe { put(&mut ManuallyDrop::take(&mut self.window_vector), target_rank, self.window_ptr); }
    }
    fn put_whole_vector(&mut self, target_rank: usize) {
        unsafe { put(&mut ManuallyDrop::take(&mut self.window_vector), target_rank, self.window_ptr); }
    }
    fn fence(&self) {
        fence(self.window_ptr);
    }
}

fn get<Buf: ?Sized>(vec: &mut Buf, target_rank: usize, window: MPI_Win) where Buf: BufferMut {
    println!("I'm inside of unsafe inside now");
    unsafe {
        ffi::MPI_Get(
            vec.pointer_mut(),
            vec.count(),
            vec.as_datatype().as_raw(),
            target_rank as c_int,
            0,
            vec.count(),
            vec.as_datatype().as_raw(),
            window
        );
    }
}
fn put<Buf: ?Sized>(vec: &mut Buf, target_rank: usize, window: MPI_Win) where Buf: BufferMut {
    unsafe {
        ffi::MPI_Put(
            vec.pointer_mut(),
            vec.count(),
            vec.as_datatype().as_raw(),
            target_rank as c_int,
            0,
            vec.count(),
            vec.as_datatype().as_raw(),
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
        println!("Drop is called");
        unsafe {
            ffi::MPI_Win_free(&mut self.window_ptr);
        }
        println!("Window has been free");
    }
}
